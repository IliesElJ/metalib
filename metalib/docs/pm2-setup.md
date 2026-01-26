# PM2 Setup Guide for MetaLib

PM2 is a production process manager that keeps your trading strategies running 24/7 with automatic restarts, logging, and monitoring.

## Prerequisites

1. **Node.js** - PM2 runs on Node.js
2. **MT5 Terminal** - Must be running before starting strategies
3. **Conda environment "adonys"** - Must be configured with all Python dependencies

## Configuration

The `ecosystem.config.js` is pre-configured for:
- **Conda root**: `C:\ProgramData\miniconda3`
- **Environment**: `adonys`
- **Python**: `C:\ProgramData\miniconda3\envs\adonys\python.exe`

If your paths differ, edit lines 15-16 in `ecosystem.config.js`:
```javascript
const CONDA_ROOT = "C:\\ProgramData\\miniconda3";
const CONDA_ENV = "adonys";
```

## Installation

### 1. Install Node.js

Download and install from: https://nodejs.org/ (LTS version recommended)

### 2. Install PM2 globally

```bash
npm install pm2 -g
```

### 3. Verify installation

```bash
pm2 --version
```

## Quick Start

### Start all strategies

```bash
cd C:\Users\Trismegist\Documents\GitHub\metalib\metalib
pm2 start ecosystem.config.js
```

Or use the helper script:
```bash
scripts\pm2-start-all.bat
```

### Check status

```bash
pm2 status
```

You should see:
```
┌─────┬──────────┬─────────────┬─────────┬─────────┬──────────┐
│ id  │ name     │ mode        │ status  │ cpu     │ memory   │
├─────┼──────────┼─────────────┼─────────┼─────────┼──────────┤
│ 0   │ metafvg  │ fork        │ online  │ 0%      │ 45.2mb   │
│ 1   │ metago   │ fork        │ online  │ 0%      │ 42.1mb   │
│ 2   │ metaob   │ fork        │ online  │ 0%      │ 43.5mb   │
│ 3   │ metaga   │ fork        │ online  │ 0%      │ 41.8mb   │
│ 4   │ metane   │ fork        │ online  │ 0%      │ 44.2mb   │
└─────┴──────────┴─────────────┴─────────┴─────────┴──────────┘
```

## Common Commands

| Command | Description |
|---------|-------------|
| `pm2 start ecosystem.config.js` | Start all strategies |
| `pm2 stop all` | Stop all strategies |
| `pm2 restart all` | Restart all strategies |
| `pm2 status` | Show status of all processes |
| `pm2 logs` | View real-time logs |
| `pm2 logs metafvg` | View logs for specific strategy |
| `pm2 monit` | Open monitoring dashboard |
| `pm2 describe metafvg` | Detailed info about a process |

## Auto-Start on Boot

To make strategies start automatically when Windows boots:

### 1. Save current process list
```bash
pm2 save
```

### 2. Generate startup script
```bash
pm2 startup
```

Follow the instructions provided by PM2.

### Alternative: Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: "At startup"
4. Action: Start a program
5. Program: `pm2`
6. Arguments: `resurrect`
7. Start in: `C:\Users\Trismegist\Documents\GitHub\metalib\metalib`

## Managing Individual Strategies

### Start specific strategy
```bash
pm2 start ecosystem.config.js --only metafvg
```

### Stop specific strategy
```bash
pm2 stop metafvg
```

### Restart specific strategy
```bash
pm2 restart metafvg
```

### Delete from PM2 (not running anymore)
```bash
pm2 delete metafvg
```

## Viewing Logs

### Real-time logs (all)
```bash
pm2 logs
```

### Real-time logs (specific)
```bash
pm2 logs metafvg
```

### Last 200 lines
```bash
pm2 logs --lines 200
```

### Log files location
- `mains/logs/metafvg-out.log`
- `mains/logs/metafvg-error.log`
- etc.

## Monitoring

### Terminal dashboard
```bash
pm2 monit
```

### Web dashboard (PM2 Plus - optional, paid service)
```bash
pm2 plus
```

## Troubleshooting

### Strategy won't start
1. Check MT5 terminal is running
2. Check logs: `pm2 logs <name>`
3. Verify Python path: `python --version`

### Strategy keeps restarting
Check error logs:
```bash
pm2 logs <name> --err --lines 100
```

Common causes:
- MT5 not connected
- Invalid config file
- Missing dependencies

### Reset everything
```bash
pm2 kill          # Stop PM2 daemon
pm2 start ecosystem.config.js   # Fresh start
```

### Check restart count
```bash
pm2 describe <name>
```

Look for `restart_time` field.

## Configuration

Edit `ecosystem.config.js` to:
- Add new strategies
- Change restart behavior
- Modify log paths
- Set environment variables

### Key settings:
```javascript
{
  max_restarts: 10,        // Max restarts before giving up
  restart_delay: 30000,    // 30 seconds between restarts
  autorestart: true,       // Auto-restart on crash
}
```

## Helper Scripts

Located in `scripts/` directory:

| Script | Description |
|--------|-------------|
| `pm2-start-all.bat` | Start all strategies |
| `pm2-stop-all.bat` | Stop all strategies |
| `pm2-restart-all.bat` | Restart all strategies |
| `pm2-status.bat` | Show status |
| `pm2-logs.bat` | View logs |
| `pm2-monitor.bat` | Open monitor dashboard |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Windows Server                        │
│                                                          │
│   MT5 Terminal (must be running)                        │
│         │                                                │
│   ┌─────┴─────────────────────────────────────────┐     │
│   │                    PM2                         │     │
│   │                                               │     │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐      │     │
│   │  │ metafvg  │ │ metago   │ │ metaob   │ ...  │     │
│   │  │ (python) │ │ (python) │ │ (python) │      │     │
│   │  └──────────┘ └──────────┘ └──────────┘      │     │
│   │                                               │     │
│   │  Auto-restart │ Logging │ Monitoring         │     │
│   └───────────────────────────────────────────────┘     │
│                                                          │
│   MetaDash (optional - can also be managed by PM2)      │
└─────────────────────────────────────────────────────────┘
```
