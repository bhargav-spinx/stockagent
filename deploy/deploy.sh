#!/usr/bin/env bash
#
# VM-side deploy step, invoked by the GitHub Actions workflow over SSH AFTER the
# code has been reset to origin/main. Idempotent and safe to run by hand too:
#
#     bash ~/stockagent/deploy/deploy.sh
#
# It does NOT pull — the workflow already fast-forwards the working tree. This
# script only: (re)installs deps when requirements changed, restarts the
# systemd service, and verifies it came back up.
#
# Requirements on the VM (one-time, see deploy/README.md):
#   • venv at ~/stockagent/venv
#   • passwordless sudo for: systemctl restart stockbot
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_DIR/venv"
SERVICE="stockbot"
cd "$REPO_DIR"

log() { echo "[deploy $(date '+%H:%M:%S')] $*"; }

log "deploying $(git rev-parse --short HEAD) on $(hostname)"

# --- install deps only if requirements.txt changed in this deploy -------------
# ORIG_HEAD is the pre-reset commit (set by the workflow's `git reset --hard`).
NEED_PIP=1
if git rev-parse --verify -q ORIG_HEAD >/dev/null; then
  if git diff --quiet ORIG_HEAD HEAD -- requirements.txt; then
    NEED_PIP=0
  fi
fi
if [ "$NEED_PIP" = "1" ]; then
  log "requirements.txt changed (or unknown) -> pip install"
  "$VENV/bin/pip" install -r requirements.txt
else
  log "requirements.txt unchanged -> skipping pip install"
fi

# --- restart + health check ---------------------------------------------------
log "restarting $SERVICE"
sudo systemctl restart "$SERVICE"
sleep 4

if systemctl is-active --quiet "$SERVICE"; then
  log "OK — $SERVICE active at $(git rev-parse --short HEAD)"
else
  log "FAILED — $SERVICE did not come back up:"
  systemctl status "$SERVICE" --no-pager -l 2>&1 | tail -25 || true
  exit 1
fi
