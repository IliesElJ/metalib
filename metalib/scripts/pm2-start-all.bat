@echo off
echo Starting all MetaLib strategies with PM2...
cd /d "%~dp0.."
pm2 start ecosystem.config.js
pm2 save
echo.
echo All strategies started. Use 'pm2 status' to check.
pause
