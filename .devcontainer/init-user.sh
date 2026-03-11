#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${UBUNTU_PASSWORD:-}" ]]; then
    echo "ubuntu:${UBUNTU_PASSWORD}" | chpasswd
else
    echo "UBUNTU_PASSWORD is not set; sudo will keep asking for a password that does not exist." >&2
fi

exec "$@"
