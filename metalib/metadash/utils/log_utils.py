"""
Log File Utilities
Functions for handling and parsing log files
"""
import os
import re
from datetime import datetime
from pathlib import Path


def get_logs_directory():
    """
    Get the path to the logs directory.

    Returns:
        Path object pointing to the logs directory
    """
    # Assuming metadash is in metalib/metadash and logs are in metalib/mains/logs
    current_dir = Path(__file__).parent
    metalib_root = current_dir.parent.parent
    logs_dir = metalib_root / "mains" / "logs"
    return logs_dir


def get_available_log_files():
    """
    Get list of all available log files.

    Returns:
        List of log file names
    """
    logs_dir = get_logs_directory()
    if not logs_dir.exists():
        return []

    log_files = [f.name for f in logs_dir.glob("output_*.log")]
    return sorted(log_files, reverse=True)


def parse_log_filename(filename):
    """
    Parse log filename to extract strategy instance and date.

    Args:
        filename: Log filename (e.g., "output_metafvg_audjpy_2025-12-03.log")

    Returns:
        Tuple of (strategy_instance, date_str) or (None, None) if parsing fails
    """
    pattern = r"output_(.+?)_(\d{4}-\d{2}-\d{2})\.log"
    match = re.match(pattern, filename)
    if match:
        strategy_instance = match.group(1)
        date_str = match.group(2)
        return strategy_instance, date_str
    return None, None


def get_strategy_instances():
    """
    Get unique list of strategy instances from log files.

    Returns:
        Sorted list of strategy instance names
    """
    log_files = get_available_log_files()
    instances = set()

    for filename in log_files:
        instance, _ = parse_log_filename(filename)
        if instance:
            instances.add(instance)

    return sorted(instances)


def get_dates_for_strategy(strategy_instance):
    """
    Get available dates for a specific strategy instance.

    Args:
        strategy_instance: Strategy instance name (e.g., "metafvg_audjpy")

    Returns:
        Sorted list of date strings (newest first)
    """
    log_files = get_available_log_files()
    dates = []

    for filename in log_files:
        instance, date_str = parse_log_filename(filename)
        if instance == strategy_instance:
            dates.append(date_str)

    return sorted(dates, reverse=True)


def read_log_file(strategy_instance, date_str):
    """
    Read and return the contents of a log file.

    Args:
        strategy_instance: Strategy instance name
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Log file contents as string, or None if file not found
    """
    logs_dir = get_logs_directory()
    filename = f"output_{strategy_instance}_{date_str}.log"
    filepath = logs_dir / filename

    if not filepath.exists():
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading log file: {str(e)}"


def parse_log_content(log_content):
    """
    Parse log content and structure it for display.

    Args:
        log_content: Raw log file content

    Returns:
        List of log entry dictionaries with timestamp, level, and message
    """
    if not log_content:
        return []

    entries = []
    lines = log_content.split("\n")

    current_timestamp = None
    for line in lines:
        if not line.strip():
            continue

        # Check if line starts with "Last time in the index:"
        if line.startswith("Last time in the index:"):
            timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if timestamp_match:
                current_timestamp = timestamp_match.group(1)
                entries.append({
                    "timestamp": current_timestamp,
                    "level": "INFO",
                    "message": line,
                    "type": "timestamp_marker"
                })
        else:
            # Regular log line
            level = "INFO"
            if "error" in line.lower() or "failed" in line.lower():
                level = "ERROR"
            elif "warning" in line.lower():
                level = "WARNING"
            elif "success" in line.lower() or "entered" in line.lower():
                level = "SUCCESS"

            entries.append({
                "timestamp": current_timestamp or "",
                "level": level,
                "message": line,
                "type": "log_entry"
            })

    return entries


def get_log_statistics(log_content):
    """
    Calculate statistics from log content.

    Args:
        log_content: Raw log file content

    Returns:
        Dictionary with log statistics
    """
    if not log_content:
        return {
            "total_lines": 0,
            "timestamp_markers": 0,
            "errors": 0,
            "warnings": 0
        }

    lines = log_content.split("\n")
    total_lines = len([l for l in lines if l.strip()])

    timestamp_markers = len([l for l in lines if "Last time in the index:" in l])
    errors = len([l for l in lines if "error" in l.lower() or "failed" in l.lower()])
    warnings = len([l for l in lines if "warning" in l.lower()])

    return {
        "total_lines": total_lines,
        "timestamp_markers": timestamp_markers,
        "errors": errors,
        "warnings": warnings
    }
