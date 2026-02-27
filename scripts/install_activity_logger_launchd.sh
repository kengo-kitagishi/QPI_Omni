#!/usr/bin/env bash
set -euo pipefail

LABEL="com.kitak.qpi.activity-logger"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
REPO_ROOT="/Users/kitak/QPI_Omni"
SCRIPT_PATH="$REPO_ROOT/scripts/session_activity_logger.py"
PYTHON_BIN="/usr/bin/python3"
INTERVAL="${1:-900}"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$REPO_ROOT/.figure_history"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON_BIN}</string>
    <string>${SCRIPT_PATH}</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>

  <key>StartInterval</key>
  <integer>${INTERVAL}</integer>

  <key>RunAtLoad</key>
  <true/>

  <key>StandardOutPath</key>
  <string>${REPO_ROOT}/.figure_history/activity_logger.out.log</string>

  <key>StandardErrorPath</key>
  <string>${REPO_ROOT}/.figure_history/activity_logger.err.log</string>
</dict>
</plist>
EOF

if launchctl list | grep -q "${LABEL}"; then
  launchctl unload "$PLIST" >/dev/null 2>&1 || true
fi

launchctl load "$PLIST"
launchctl start "$LABEL" || true

echo "Installed: $PLIST"
echo "Label: $LABEL"
echo "Interval(seconds): $INTERVAL"
echo "Logs: $REPO_ROOT/.figure_history/activity_logger.out.log"
