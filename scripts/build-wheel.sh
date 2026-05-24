#!/usr/bin/env bash
#
# Build the `dagnam` wheel + sdist into dag-lib/dist/.
#
# Two uses:
#   1. Release prep — produces the exact artifacts you twine-upload to PyPI
#      (see RELEASE.md).
#   2. Local training tests — mvp-backend's training pipeline, when
#      DAGNAM_PACKAGE_SOURCE=wheelhouse, installs dagnam from this dist/ via
#      `uv pip install --find-links <dag-lib>/dist`. That exercises the same
#      artifact you'll publish, before you publish it.
#
# Usage:
#   ./scripts/build-wheel.sh             # clean build into dist/ (default)
#   ./scripts/build-wheel.sh --no-clean  # keep existing dist/ contents
#
set -euo pipefail

no_clean=0
if [[ "${1:-}" == "--no-clean" ]]; then
    no_clean=1
fi

# scripts/ lives directly under the dag-lib root.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
daglib_root="$(dirname "$script_dir")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not on PATH. Install it: https://docs.astral.sh/uv/" >&2
    exit 1
fi

dist_dir="$daglib_root/dist"
if [[ "$no_clean" -eq 0 && -d "$dist_dir" ]]; then
    echo "Cleaning $dist_dir"
    rm -rf "$dist_dir"
fi

echo "Building dagnam (wheel + sdist) into dist/ ..."
( cd "$daglib_root" && uv build )

echo
echo "Built: $(ls "$dist_dir"/dagnam-*.whl 2>/dev/null | head -n1)"
echo
echo "To exercise it from the training pipeline, set in mvp-backend/.env:"
echo "  DAGNAM_PACKAGE_SOURCE=wheelhouse"
echo "  DAGNAM_LOCAL_PATH=$daglib_root"
