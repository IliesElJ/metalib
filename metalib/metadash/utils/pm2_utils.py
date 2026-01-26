"""
PM2 Utilities Module
Handles PM2 process management for trading strategies.
"""

import subprocess
import json
from typing import List, Dict, Optional


def get_pm2_status() -> List[Dict]:
    """
    Get status of all PM2 processes.

    Returns:
        List of process info dicts with keys:
        - name: process name
        - status: 'online', 'stopped', 'errored', etc.
        - cpu: CPU usage percentage
        - memory: Memory usage in MB
        - uptime: Uptime in milliseconds
        - restarts: Number of restarts
    """
    try:
        # Use shell=True on Windows for better compatibility
        result = subprocess.run(
            "pm2 jlist",
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
            encoding="utf-8",
            errors="ignore",
        )

        if result.returncode != 0:
            return []

        output = result.stdout.strip()
        if not output:
            return []

        # Try to find valid JSON array in output (skip any non-JSON prefix)
        start_idx = output.find("[")
        end_idx = output.rfind("]")
        if start_idx == -1 or end_idx == -1:
            return []

        json_str = output[start_idx : end_idx + 1]
        processes = json.loads(json_str)

        status_list = []
        for proc in processes:
            pm2_env = proc.get("pm2_env", {})
            monit = proc.get("monit", {})

            # Calculate uptime string
            uptime_ms = pm2_env.get("pm_uptime", 0)
            uptime_str = _format_uptime(uptime_ms)

            status_list.append(
                {
                    "name": proc.get("name", "unknown"),
                    "pm_id": proc.get("pm_id", -1),
                    "status": pm2_env.get("status", "unknown"),
                    "cpu": monit.get("cpu", 0),
                    "memory": round(
                        monit.get("memory", 0) / (1024 * 1024), 1
                    ),  # Convert to MB
                    "uptime": uptime_str,
                    "uptime_ms": uptime_ms,
                    "restarts": pm2_env.get("restart_time", 0),
                    "pid": proc.get("pid", None),
                }
            )

        return status_list

    except subprocess.TimeoutExpired:
        return []
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"PM2 JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"PM2 status error: {e}")
        return []


def _format_uptime(uptime_ms: int) -> str:
    """Format uptime from milliseconds to human-readable string."""
    if not uptime_ms:
        return "N/A"

    import time
    from datetime import datetime

    # pm_uptime is the timestamp when the process started
    try:
        start_time = datetime.fromtimestamp(uptime_ms / 1000)
        now = datetime.now()
        delta = now - start_time

        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes}m"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            return f"{days}d {hours}h"
    except:
        return "N/A"


def pm2_start(name: Optional[str] = None) -> Dict:
    """
    Start PM2 process(es).

    Args:
        name: Process name to start, or None to start all

    Returns:
        Dict with 'success' bool and 'message' string
    """
    try:
        if name:
            cmd = f"pm2 start {name}"
        else:
            cmd = "pm2 start ecosystem.config.js"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True,
            cwd=_get_project_root(),
            encoding="utf-8",
            errors="ignore",
        )

        if result.returncode == 0:
            return {"success": True, "message": f"Started {name or 'all processes'}"}
        else:
            return {"success": False, "message": result.stderr or "Failed to start"}

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Command timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def pm2_stop(name: Optional[str] = None) -> Dict:
    """
    Stop PM2 process(es).

    Args:
        name: Process name to stop, or None to stop all

    Returns:
        Dict with 'success' bool and 'message' string
    """
    try:
        if name:
            cmd = f"pm2 stop {name}"
        else:
            cmd = "pm2 stop all"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True,
            encoding="utf-8",
            errors="ignore",
        )

        if result.returncode == 0:
            return {"success": True, "message": f"Stopped {name or 'all processes'}"}
        else:
            return {"success": False, "message": result.stderr or "Failed to stop"}

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Command timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def pm2_restart(name: Optional[str] = None) -> Dict:
    """
    Restart PM2 process(es).

    Args:
        name: Process name to restart, or None to restart all

    Returns:
        Dict with 'success' bool and 'message' string
    """
    try:
        if name:
            cmd = f"pm2 restart {name}"
        else:
            cmd = "pm2 restart all"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True,
            encoding="utf-8",
            errors="ignore",
        )

        if result.returncode == 0:
            return {"success": True, "message": f"Restarted {name or 'all processes'}"}
        else:
            return {"success": False, "message": result.stderr or "Failed to restart"}

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Command timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def pm2_save() -> Dict:
    """
    Save PM2 process list for auto-restart on boot.

    Returns:
        Dict with 'success' bool and 'message' string
    """
    try:
        result = subprocess.run(
            "pm2 save",
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
            encoding="utf-8",
            errors="ignore",
        )

        if result.returncode == 0:
            return {"success": True, "message": "Process list saved"}
        else:
            return {"success": False, "message": result.stderr or "Failed to save"}

    except Exception as e:
        return {"success": False, "message": str(e)}


def is_pm2_available() -> bool:
    """Check if PM2 is installed and available."""
    try:
        result = subprocess.run(
            "pm2 --version",
            capture_output=True,
            text=True,
            timeout=5,
            shell=True,
            encoding="utf-8",
            errors="ignore",
        )
        return result.returncode == 0
    except:
        return False


def get_pm2_summary() -> Dict:
    """
    Get summary of PM2 processes.

    Returns:
        Dict with counts: total, online, stopped, errored
    """
    processes = get_pm2_status()

    summary = {
        "total": len(processes),
        "online": sum(1 for p in processes if p["status"] == "online"),
        "stopped": sum(1 for p in processes if p["status"] == "stopped"),
        "errored": sum(1 for p in processes if p["status"] == "errored"),
        "available": is_pm2_available(),
    }

    return summary


def _get_project_root() -> str:
    """Get the project root directory."""
    import os

    # Go up from metadash/utils to metalib root
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current))
