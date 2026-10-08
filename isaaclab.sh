#!/usr/bin/env bash
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# SimVLA compatibility entry point for scripts using `isaaclab.sh -p`.
set -euo pipefail
simvla_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:---help}" in
  -p|--python)
    shift
    export SIMVLA_REPO_ROOT="$simvla_root"
    export PYTHONPATH="$simvla_root/source/isaaclab:$simvla_root/source/isaaclab_assets:$simvla_root/source/isaaclab_tasks:$simvla_root/source/isaaclab_rl${PYTHONPATH:+:$PYTHONPATH}"
    exec python "$@"
    ;;
  -h|--help)
    echo 'Usage: ./isaaclab.sh -p SCRIPT [ARGS...]'
    echo 'Activate your configured simulator environment first.'
    echo 'For supported workflows use simvla run; setup: docs/simulation.md'
    ;;
  *)
    echo 'This SimVLA release supports -p/--python only; see docs/simulation.md.' >&2
    exit 2
    ;;
esac
