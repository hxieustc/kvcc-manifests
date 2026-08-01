#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failures=0

pass() {
  printf 'PASS: %s\n' "$1"
}

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  failures=$((failures + 1))
}

assert_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    pass "$path exists"
  else
    fail "$path is missing"
  fi
}

assert_contains() {
  local path="$1"
  local pattern="$2"
  if [[ ! -f "$path" ]]; then
    fail "$path is missing (wanted: $pattern)"
  elif grep -Eq -- "$pattern" "$path"; then
    pass "$path contains: $pattern"
  else
    fail "$path lacks: $pattern"
  fi
}

assert_not_contains() {
  local path="$1"
  local pattern="$2"
  if [[ ! -f "$path" ]]; then
    fail "$path is missing (cannot inspect: $pattern)"
  elif grep -Eq -- "$pattern" "$path"; then
    fail "$path contains forbidden pattern: $pattern"
  else
    pass "$path omits: $pattern"
  fi
}

assert_file kubernetes/dev-pod.yaml

if [[ "$failures" -ne 0 ]]; then
  printf '\n%d repository contract failure(s)\n' "$failures" >&2
  exit 1
fi

printf '\nRepository contract checks passed\n'
