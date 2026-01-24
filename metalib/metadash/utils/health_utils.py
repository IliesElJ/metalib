"""
Health Monitoring Utilities
Functions for checking strategy health status based on log activity and config timeframes.
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import yaml


def get_config_directory():
    """Get the path to the config/prod directory."""
    current_dir = Path(__file__).parent
    metalib_root = current_dir.parent.parent
    return metalib_root / "config" / "prod"


def get_logs_directory():
    """Get the path to the logs directory."""
    current_dir = Path(__file__).parent
    metalib_root = current_dir.parent.parent
    return metalib_root / "mains" / "logs"


def parse_timeframe(timeframe_str):
    """
    Parse MT5 timeframe string to minutes.

    Args:
        timeframe_str: e.g., 'mt5.TIMEFRAME_M1', 'mt5.TIMEFRAME_H1'

    Returns:
        Number of minutes for the timeframe
    """
    if not timeframe_str:
        return 1  # Default to 1 minute

    timeframe_map = {
        "mt5.TIMEFRAME_M1": 1,
        "mt5.TIMEFRAME_M5": 5,
        "mt5.TIMEFRAME_M15": 15,
        "mt5.TIMEFRAME_M30": 30,
        "mt5.TIMEFRAME_H1": 60,
        "mt5.TIMEFRAME_H4": 240,
        "mt5.TIMEFRAME_D1": 1440,
    }

    return timeframe_map.get(timeframe_str, 1)


def get_staleness_thresholds(timeframe_minutes):
    """
    Get staleness thresholds based on timeframe.

    Args:
        timeframe_minutes: Timeframe in minutes

    Returns:
        Tuple of (stale_threshold_minutes, stopped_threshold_minutes)
    """
    # Stale = 10x the timeframe (or minimum 10 min)
    # Stopped = 30x the timeframe (or minimum 30 min)
    stale = max(10, timeframe_minutes * 10)
    stopped = max(30, timeframe_minutes * 30)

    return stale, stopped


def load_all_configs():
    """
    Load all production config files and build a mapping of tag -> config info.

    Returns:
        Dict mapping strategy tag to config info including timeframe
    """
    config_dir = get_config_directory()
    tag_configs = {}

    if not config_dir.exists():
        return tag_configs

    for config_file in config_dir.glob("*.yaml"):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config:
                continue

            for instance_name, instance_config in config.items():
                if isinstance(instance_config, dict):
                    tag = instance_config.get("tag")
                    if tag:
                        timeframe_str = instance_config.get("timeframe", "mt5.TIMEFRAME_M1")
                        timeframe_minutes = parse_timeframe(timeframe_str)
                        stale_thresh, stopped_thresh = get_staleness_thresholds(timeframe_minutes)

                        tag_configs[tag] = {
                            "strategy_type": instance_config.get("strategy_type", "unknown"),
                            "symbols": instance_config.get("symbols", []),
                            "timeframe_str": timeframe_str,
                            "timeframe_minutes": timeframe_minutes,
                            "stale_threshold": stale_thresh,
                            "stopped_threshold": stopped_thresh,
                            "config_file": config_file.name,
                            "instance_name": instance_name,
                        }
        except Exception as e:
            print(f"Error loading config {config_file}: {e}")
            continue

    return tag_configs


def get_last_log_timestamp(strategy_tag):
    """
    Get the last timestamp from a strategy's most recent log file.

    Args:
        strategy_tag: Strategy tag (e.g., 'metafvg_dax')

    Returns:
        Tuple of (datetime or None, log_date_str or None)
    """
    logs_dir = get_logs_directory()

    if not logs_dir.exists():
        return None, None

    # Find log files for this strategy, sorted by date (newest first)
    pattern = f"output_{strategy_tag}_*.log"
    log_files = sorted(logs_dir.glob(pattern), reverse=True)

    if not log_files:
        return None, None

    # Get the most recent log file
    latest_log = log_files[0]

    # Extract date from filename
    match = re.search(r"(\d{4}-\d{2}-\d{2})\.log$", latest_log.name)
    log_date_str = match.group(1) if match else None

    # Read the file and find the last timestamp
    try:
        with open(latest_log, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all timestamps in the log
        timestamps = re.findall(r"Last time in the index: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content)

        if timestamps:
            last_ts_str = timestamps[-1]
            last_ts = datetime.strptime(last_ts_str, "%Y-%m-%d %H:%M:%S")
            return last_ts, log_date_str

        # If no timestamps found, use file modification time
        mtime = os.path.getmtime(latest_log)
        return datetime.fromtimestamp(mtime), log_date_str

    except Exception as e:
        print(f"Error reading log {latest_log}: {e}")
        return None, None


def is_weekend():
    """Check if today is a weekend (Saturday=5, Sunday=6)."""
    return datetime.now().weekday() >= 5


def is_market_closed_for_symbol(symbols):
    """
    Check if market is likely closed for given symbols.
    This is a simplified check - forex markets are closed on weekends,
    indices have specific trading hours.

    Args:
        symbols: List of symbol names

    Returns:
        Boolean indicating if market is likely closed
    """
    # For now, just check for weekends
    # Could be enhanced with proper market hours later
    return is_weekend()


def get_strategy_status(strategy_tag, config_info, now=None):
    """
    Get the health status for a strategy.

    Args:
        strategy_tag: Strategy tag
        config_info: Config info dict for this strategy
        now: Current datetime (optional, for testing)

    Returns:
        Dict with status information
    """
    if now is None:
        now = datetime.now()

    last_ts, log_date = get_last_log_timestamp(strategy_tag)

    symbols = config_info.get("symbols", [])
    market_closed = is_market_closed_for_symbol(symbols)
    weekend = is_weekend()

    if last_ts is None:
        return {
            "tag": strategy_tag,
            "status": "unknown",
            "status_color": "secondary",
            "last_activity": None,
            "last_activity_str": "No logs found",
            "minutes_ago": None,
            "log_date": log_date,
            "market_closed": market_closed,
            "weekend": weekend,
            "timeframe": config_info.get("timeframe_str", "unknown"),
            "symbols": symbols,
            "strategy_type": config_info.get("strategy_type", "unknown"),
            "stale_threshold": config_info.get("stale_threshold", 30),
            "stopped_threshold": config_info.get("stopped_threshold", 90),
        }

    # Calculate time since last activity
    delta = now - last_ts
    minutes_ago = delta.total_seconds() / 60

    stale_threshold = config_info.get("stale_threshold", 30)
    stopped_threshold = config_info.get("stopped_threshold", 90)

    # Determine status
    if weekend:
        # On weekends, be more lenient - only mark as stopped if way overdue
        if minutes_ago > stopped_threshold * 3:
            status = "stopped"
            status_color = "danger"
        else:
            status = "weekend"
            status_color = "info"
    elif minutes_ago <= stale_threshold:
        status = "running"
        status_color = "success"
    elif minutes_ago <= stopped_threshold:
        status = "stale"
        status_color = "warning"
    else:
        status = "stopped"
        status_color = "danger"

    # Format last activity string
    if minutes_ago < 60:
        last_activity_str = f"{int(minutes_ago)} min ago"
    elif minutes_ago < 1440:
        hours = minutes_ago / 60
        last_activity_str = f"{hours:.1f} hours ago"
    else:
        days = minutes_ago / 1440
        last_activity_str = f"{days:.1f} days ago"

    return {
        "tag": strategy_tag,
        "status": status,
        "status_color": status_color,
        "last_activity": last_ts,
        "last_activity_str": last_activity_str,
        "minutes_ago": minutes_ago,
        "log_date": log_date,
        "market_closed": market_closed,
        "weekend": weekend,
        "timeframe": config_info.get("timeframe_str", "unknown"),
        "symbols": symbols,
        "strategy_type": config_info.get("strategy_type", "unknown"),
        "stale_threshold": stale_threshold,
        "stopped_threshold": stopped_threshold,
    }


def get_all_strategy_statuses():
    """
    Get health status for all configured strategies.

    Returns:
        List of status dicts, sorted by status (stopped first, then stale, then running)
    """
    configs = load_all_configs()
    statuses = []

    for tag, config_info in configs.items():
        status = get_strategy_status(tag, config_info)
        statuses.append(status)

    # Sort by status priority: stopped > stale > unknown > weekend > running
    status_priority = {
        "stopped": 0,
        "stale": 1,
        "unknown": 2,
        "weekend": 3,
        "running": 4,
    }

    statuses.sort(key=lambda x: (status_priority.get(x["status"], 5), x["tag"]))

    return statuses


def get_health_summary():
    """
    Get a summary of overall health.

    Returns:
        Dict with counts by status
    """
    statuses = get_all_strategy_statuses()

    summary = {
        "total": len(statuses),
        "running": 0,
        "stale": 0,
        "stopped": 0,
        "weekend": 0,
        "unknown": 0,
        "is_weekend": is_weekend(),
    }

    for s in statuses:
        status = s["status"]
        if status in summary:
            summary[status] += 1

    return summary
