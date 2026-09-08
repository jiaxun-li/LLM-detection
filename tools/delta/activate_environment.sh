#!/bin/bash
# Source this file; do not execute it in a separate shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source tools/delta/activate_environment.sh' >&2
    exit 2
fi

_activate_llm_detection_environment() {
    local environment_path="/projects/bhuc/${USER}/venvs/delta-smoke"
    if ! type module >/dev/null 2>&1; then
        echo 'Delta module command unavailable; use this helper on Delta.' >&2
        return 1
    fi
    module reset || return 1
    module load miniforge3-python || return 1
    if ! source "$environment_path/bin/activate"; then
        echo "Could not activate $environment_path" >&2
        return 1
    fi
    # Preserve activate/deactivate's original prompt bookkeeping. This changes
    # the label, not the installation path or any installed Python commands.
    export VIRTUAL_ENV_PROMPT='(llm-detection) '
    if [[ $- == *i* ]]; then
        PS1="${VIRTUAL_ENV_PROMPT}${_OLD_VIRTUAL_PS1-${PS1-}}"
    fi
    printf 'Environment: llm-detection\nPython: %s\n' "$(command -v python)"
}

if _activate_llm_detection_environment; then
    unset -f _activate_llm_detection_environment
else
    unset -f _activate_llm_detection_environment
    return 1
fi
