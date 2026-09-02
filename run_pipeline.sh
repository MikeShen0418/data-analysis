#!/usr/bin/env sh
set -eu
if [ -z "${SEC_USER_AGENT:-}" ]; then
  echo 'Set SEC_USER_AGENT="Your Name your.email@example.com" first.' >&2
  exit 1
fi
python ai_monitor.py run --user-agent "$SEC_USER_AGENT" "$@"
