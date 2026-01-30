"""
PM2 Utilities Module
Handles PM2 process management for trading strategies.

NOTE: The dashboard must be started from a terminal where pm2 and node are in the PATH.
"""

import subprocess
import json
import os
from typing import List, Dict, Optional


def get_pm2_status() -> List[Dict]:
    """
    Get status of all PM2 processes.

    Returns:
        List of process info dicts
    """
    try:
        result = subprocess.run(
            "pm2 jlist",
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
        )

        if result.returncode != 0:
            return []

        output = result.stdout.strip()
        if not output:
            return []

        # Find JSON array in output
        start_idx = output.find("[")
        end_idx = output.rfind("]")
        if start_idx == -1 or end_idx == -1:
            return []

        json_str = output[start_idx:end_idx + 1]
        processes = json.loads(json_str)

        status_list = []
        for proc in processes:
            pm2_env = proc.get("pm2_env", {})
            monit = proc.get("monit", {})

            uptime_ms = pm2_env.get("pm_uptime", 0)
            uptime_str = _format_uptime(uptime_ms)

            status_list.append({
                "name": proc.get("name", "unknown"),
                "pm_id": proc.get("pm_id", -1),
                "status": pm2_env.get("status", "unknown"),
                "cpu": monit.get("cpu", 0),
                "memory": round(monit.get("memory", 0) / (1024 * 1024), 1),
                "uptime": uptime_str,
                "uptime_ms": uptime_ms,
                "restarts": pm2_env.get("restart_time", 0),
                "pid": proc.get("pid", None),
            })

        return status_list

    except Exception as e:
        print(f"PM2 status error: {e}")
        return []


def _format_uptime(uptime_ms: int) -> str:
    """Format uptime from milliseconds to human-readable string."""
    if not uptime_ms:
        return "N/A"

    try:
        from datetime import datetime
        start_time = datetime.fromtimestamp(uptime_ms / 1000)
        now = datetime.now()
        delta = now - start_time
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m"
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
    """Start PM2 process(es)."""
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
        )

        if result.returncode == 0:
            return {"success": True, "message": f"Started {name or 'all processes'}"}
        else:
            return {"success": False, "message": result.stderr or "Failed to start"}

    except Exception as e:
        return {"success": False, "message": str(e)}


def pm2_stop(name: Optional[str] = None) -> Dict:
    """Stop PM2 process(es)."""
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
        )

        if result.returncode == 0:
            return {"success": True, "message": f"Stopped {name or 'all processes'}"}
        else:
            return {"success": False, "message": result.stderr or "Failed to stop"}

    except Exception as e:
        return {"success": False, "message": str(e)}


def pm2_restart(name: Optional[str] = None) -> Dict:
    """Restart PM2 process(es)."""
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
        )

        if result.returncode == 0:
            return {"success": True, "message": f"Restarted {name or 'all processes'}"}
        else:
            return {"success": False, "message": result.stderr or "Failed to restart"}

    except Exception as e:
        return {"success": False, "message": str(e)}


def pm2_save() -> Dict:
    """Save PM2 process list."""
    try:
        result = subprocess.run(
            "pm2 save",
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
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
        )
        return result.returncode == 0
    except:
        return False


def get_pm2_summary() -> Dict:
    """Get summary of PM2 processes."""
    processes = get_pm2_status()

    return {
        "total": len(processes),
        "online": sum(1 for p in processes if p["status"] == "online"),
        "stopped": sum(1 for p in processes if p["status"] == "stopped"),
        "errored": sum(1 for p in processes if p["status"] == "errored"),
        "available": is_pm2_available(),
    }


def _get_project_root() -> str:
    """Get the project root directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current))
