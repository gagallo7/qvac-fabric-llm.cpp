#!/usr/bin/env bash
#
# Create (once) and reuse a uv venv for the qvac bench scripts.
#
# The venv is anchored to this repo, not to $PWD, so the script can be invoked
# from any directory -- or through a symlink -- and always lands in the same
# environment:
#
#   scripts/qvac-bench-env.sh                     # create/update, print the path
#   scripts/qvac-bench-env.sh python scripts/qvac-bench.py -c scripts/llama-bench.json
#   source scripts/qvac-bench-env.sh              # activate in the current shell
#
# Repeat invocations are a no-op: the installed requirement files are hashed
# into a stamp inside the venv, so uv only runs when that hash changes (or with
# --force / --upgrade).
#
# Env overrides:
#   QVAC_BENCH_VENV    venv location (default: <repo>/.venv, already gitignored)
#   QVAC_BENCH_PYTHON  interpreter for `uv venv --python` (default: uv's pick)

# --------------------------------------------------------------------------- #
# Sourced mode: do the work in a child shell (keeps `set -e` out of the caller's
# shell), then activate the venv it reports back.
# --------------------------------------------------------------------------- #
if [ -n "${ZSH_VERSION:-}" ]; then
    # zsh-only expansion; Bash skips this branch.
    # shellcheck disable=SC2296
    __qvac_self="${(%):-%x}"
    case "$ZSH_EVAL_CONTEXT" in *:file) __qvac_sourced=1 ;; *) __qvac_sourced=0 ;; esac
else
    __qvac_self="${BASH_SOURCE[0]}"
    [ "$__qvac_self" != "$0" ] && __qvac_sourced=1 || __qvac_sourced=0
fi

if [ "$__qvac_sourced" = 1 ]; then
    __qvac_venv="$(bash "$__qvac_self" --print-venv)" || return 1
    # shellcheck disable=SC1091
    . "$__qvac_venv/bin/activate"
    echo "activated $__qvac_venv" >&2
    unset __qvac_self __qvac_sourced __qvac_venv
    return 0
fi
unset __qvac_self __qvac_sourced

set -euo pipefail

# --------------------------------------------------------------------------- #
# Locate the repo through symlinks (readlink -f is absent on older macOS)
# --------------------------------------------------------------------------- #
self="$0"
while [ -L "$self" ]; do
    self_dir="$(cd -P "$(dirname "$self")" && pwd)"
    self="$(readlink "$self")"
    case "$self" in
        /*) ;;
        *) self="$self_dir/$self" ;;
    esac
done
SCRIPT_DIR="$(cd -P "$(dirname "$self")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

VENV="${QVAC_BENCH_VENV:-$REPO_ROOT/.venv}"
REQS=("$REPO_ROOT/requirements/requirements-qvac-bench.txt")
PYTHON_SPEC="${QVAC_BENCH_PYTHON:-}"
MIN_PYTHON="3.10"   # qvac-bench.py uses PEP 604 annotations at runtime

force=0
upgrade=0
print_venv=0
cmd=()

usage() {
    cat >&2 <<EOF
usage: qvac-bench-env.sh [options] [--] [command ...]

  -r, --reqs FILE   extra requirements file (repeatable)
  -p, --python VER  interpreter for a fresh venv, e.g. 3.12
  -f, --force       recreate the venv from scratch
  -U, --upgrade     let uv upgrade already-installed packages
      --print-venv  print the venv path on stdout and exit
  -h, --help        this message

With a command, it runs inside the venv; without one, the venv path is printed.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -r|--reqs)    REQS+=("$2"); shift 2 ;;
        -p|--python)  PYTHON_SPEC="$2"; shift 2 ;;
        -f|--force)   force=1; shift ;;
        -U|--upgrade) upgrade=1; shift ;;
        --print-venv) print_venv=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; cmd=("$@"); break ;;
        -*)           echo "unknown option: $1" >&2; usage; exit 2 ;;
        *)            cmd=("$@"); break ;;
    esac
done

log() { echo "[bench-env] $*" >&2; }

if ! command -v uv >/dev/null 2>&1; then
    log "uv not found. Install it with:"
    log "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    log "(or: pipx install uv / brew install uv), then re-run this script."
    exit 127
fi

for req in "${REQS[@]}"; do
    [ -f "$req" ] || { log "missing requirements file: $req"; exit 1; }
done

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | cut -d' ' -f1
    else
        shasum -a 256 | cut -d' ' -f1
    fi
}

# The stamp covers which files were installed and their contents, so editing a
# requirements file (or adding one with -r) triggers exactly one reinstall.
stamp_file="$VENV/.qvac-bench-reqs"
stamp="$( { printf '%s\n' "${REQS[@]}"; cat "${REQS[@]}"; } | sha256 )"

if [ "$force" = 1 ] && [ -d "$VENV" ]; then
    log "removing $VENV"
    rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
    log "creating venv at $VENV"
    if [ -n "$PYTHON_SPEC" ]; then
        uv venv --python "$PYTHON_SPEC" "$VENV" >&2
    else
        uv venv "$VENV" >&2
    fi
    rm -f "$stamp_file"
fi

"$VENV/bin/python" - "$MIN_PYTHON" <<'PY' >&2
import sys
want = tuple(int(p) for p in sys.argv[1].split("."))
if sys.version_info[:len(want)] < want:
    have = ".".join(str(p) for p in sys.version_info[:3])
    sys.exit(f"[bench-env] venv python is {have}, the bench scripts need "
             f">= {sys.argv[1]}; re-run with --force --python {sys.argv[1]}")
PY

if [ "$upgrade" = 1 ] || [ ! -f "$stamp_file" ] || [ "$(cat "$stamp_file")" != "$stamp" ]; then
    req_args=()
    for req in "${REQS[@]}"; do
        req_args+=(-r "$req")
    done
    log "installing requirements"
    if [ "$upgrade" = 1 ]; then
        uv pip install --python "$VENV/bin/python" --upgrade "${req_args[@]}" >&2
    else
        uv pip install --python "$VENV/bin/python" "${req_args[@]}" >&2
    fi
    echo "$stamp" > "$stamp_file"
else
    log "requirements already installed (stamp $(echo "$stamp" | cut -c1-12))"
fi

if [ "$print_venv" = 1 ]; then
    echo "$VENV"
    exit 0
fi

if [ ${#cmd[@]} -eq 0 ]; then
    log "venv ready: $VENV"
    log "run:      $0 python scripts/qvac-bench.py -c scripts/llama-bench.json"
    log "activate: source ${self}"
    exit 0
fi

export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME
exec "${cmd[@]}"
