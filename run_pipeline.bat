@echo off
if "%SEC_USER_AGENT%"=="" (
  echo Please set SEC_USER_AGENT first, for example:
  echo set SEC_USER_AGENT=Your Name your.email@example.com
  exit /b 1
)
python ai_monitor.py run --user-agent "%SEC_USER_AGENT%" %*
