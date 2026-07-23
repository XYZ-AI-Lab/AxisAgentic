#!/usr/bin/env bash
# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AXIS_INSTALL_EXTRAS="${AXIS_INSTALL_EXTRAS:-dev}"

"${PYTHON_BIN}" - <<'PY'
import sys

required = (3, 12)
if sys.version_info < required:
    version = ".".join(str(part) for part in sys.version_info[:3])
    raise SystemExit(f"AxisAgentic requires Python 3.12 or newer; {version} is active.")
PY

export AXIS_DATA_DIR="${AXIS_DATA_DIR:-${PROJECT_ROOT}/data}"
export AXIS_MODEL_DIR="${AXIS_MODEL_DIR:-${PROJECT_ROOT}/models}"
export AXIS_LOG_DIR="${AXIS_LOG_DIR:-${PROJECT_ROOT}/logs}"

mkdir -p "${AXIS_DATA_DIR}" "${AXIS_MODEL_DIR}" "${AXIS_LOG_DIR}" "${PROJECT_ROOT}/.envs"

install_target="${PROJECT_ROOT}"
if [[ -n "${AXIS_INSTALL_EXTRAS}" ]]; then
    install_target="${PROJECT_ROOT}[${AXIS_INSTALL_EXTRAS}]"
fi
"${PYTHON_BIN}" -m pip install -e "${install_target}"

env_file="${PROJECT_ROOT}/.envs/axis_agentic_env.sh"
{
    printf 'export AXIS_DATA_DIR=%q\n' "${AXIS_DATA_DIR}"
    printf 'export AXIS_MODEL_DIR=%q\n' "${AXIS_MODEL_DIR}"
    printf 'export AXIS_LOG_DIR=%q\n' "${AXIS_LOG_DIR}"
} > "${env_file}"

printf 'AxisAgentic installed from %s\n' "${PROJECT_ROOT}"
printf 'Environment file written to %s\n' "${env_file}"
printf 'Activate it with: source %q\n' "${env_file}"
