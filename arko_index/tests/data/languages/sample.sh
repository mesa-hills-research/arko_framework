#!/usr/bin/env bash
# Deploys the service: builds, uploads and restarts.
set -euo pipefail

readonly ROOT="$(cd "$(dirname "$0")" && pwd)"
VERBOSE=${VERBOSE:-0}

# Prints a message when VERBOSE is set.
log() {
  if [ "$VERBOSE" = 1 ]; then
    echo "[deploy] $*" >&2
  fi
}

# Builds the release binary.
build() {
  log "building in $ROOT"
  cargo build --release
  strip target/release/service
}

function upload {
  local host=$1
  for file in target/release/service config/*.toml; do
    scp "$file" "$host:/srv/service/"
  done
}

restart() {
  ssh "$1" 'systemctl restart service' || {
    echo "restart failed on $1" >&2
    return 1
  }
}

case "${1:-all}" in
  build) build ;;
  upload) upload "$2" ;;
  all)
    build
    upload "$2"
    restart "$2"
    ;;
  *) echo "usage: $0 build|upload|all HOST" >&2; exit 2 ;;
esac

cat <<EOF
done: deployed to ${2:-nowhere}
EOF
