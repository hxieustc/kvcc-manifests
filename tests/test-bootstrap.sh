#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT

mkdir -p \
  "$FIXTURE/workspace/manifests/scripts" \
  "$FIXTURE/workspace/vllm" \
  "$FIXTURE/bin" \
  "$FIXTURE/home"
cp "$REPO_ROOT/scripts/bootstrap.sh" \
  "$FIXTURE/workspace/manifests/scripts/bootstrap.sh"

CALL_LOG="$FIXTURE/calls.log"
export CALL_LOG

write_stub() {
  local name="$1"
  shift
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' "$@" \
    >"$FIXTURE/bin/$name"
  chmod +x "$FIXTURE/bin/$name"
}

write_stub dpkg \
  'if [[ "${*: -1}" == "build-essential" ]]; then exit 1; fi' \
  'exit 0'
write_stub apt-get 'printf "apt-get %s\n" "$*" >>"$CALL_LOG"'
write_stub repo 'exit 0'
write_stub rustc 'exit 0'
write_stub curl 'printf "curl %s\n" "$*" >>"$CALL_LOG"'
write_stub uv \
  'printf "uv %s\n" "$*" >>"$CALL_LOG"' \
  'if [[ "${1:-}" == "venv" ]]; then' \
  '  venv="${*: -1}"' \
  '  mkdir -p "$venv/bin"' \
  '  printf "#!/usr/bin/env bash\nprintf '\''Python 3.12.3\\n'\''\n" >"$venv/bin/python"' \
  '  chmod +x "$venv/bin/python"' \
  'fi'
write_stub pre-commit \
  'printf "pre-commit %s\n" "$*" >>"$CALL_LOG"'

run_bootstrap() {
  env \
    HOME="$FIXTURE/home" \
    PATH="$FIXTURE/bin:/usr/bin:/bin" \
    KVCC_EFFECTIVE_UID=0 \
    "$@" \
    bash "$FIXTURE/workspace/manifests/scripts/bootstrap.sh"
}

run_bootstrap

grep -F 'apt-get update' "$CALL_LOG" >/dev/null
grep -F \
  'apt-get install -y --no-install-recommends build-essential' \
  "$CALL_LOG" >/dev/null
grep -F 'uv venv --python 3.12' "$CALL_LOG" >/dev/null
grep -F 'maturin[patchelf]' "$CALL_LOG" >/dev/null
grep -F 'nixl' "$CALL_LOG" >/dev/null
if grep -F 'pre-commit install' "$CALL_LOG" >/dev/null; then
  echo 'pre-commit hooks must be opt-in' >&2
  exit 1
fi

: >"$CALL_LOG"
run_bootstrap KVCC_INSTALL_PRECOMMIT=1
grep -F 'pre-commit install' "$CALL_LOG" >/dev/null

echo 'PASS: bootstrap installs missing prerequisites and creates an isolated venv'
