#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DESTINATION="${BOXFUSION_TR3D_OFFICIAL_CHECKPOINT:-$ROOT/models/tr3d_1xb16_scannet-3d-18class.pth}"
URL="https://download.openmmlab.com/mmdetection3d/v1.1.0_models/tr3d/tr3d_1xb16_scannet-3d-18class/tr3d_1xb16_scannet-3d-18class.pth"
EXPECTED_SHA256="c3e9435c3a22b49e0e57b0a619003dd0f18bef484b97173537752fb6fa30298f"

verify() {
  local observed
  observed="$(sha256sum "$1" | awk '{print $1}')"
  [[ "$observed" == "$EXPECTED_SHA256" ]] || {
    echo "TR3D checkpoint SHA256 mismatch: $observed" >&2
    return 1
  }
}

if [[ -f "$DESTINATION" ]]; then
  verify "$DESTINATION"
  echo "Official TR3D checkpoint already verified: $DESTINATION"
  exit 0
fi
[[ ! -e "$DESTINATION" ]] || {
  echo "Refusing non-file checkpoint destination: $DESTINATION" >&2
  exit 2
}

mkdir -p "$(dirname "$DESTINATION")"
temporary="$(mktemp "$(dirname "$DESTINATION")/.tr3d-checkpoint.XXXXXX")"
cleanup() {
  rm -f -- "$temporary"
}
trap cleanup EXIT
wget --https-only --output-document="$temporary" "$URL"
verify "$temporary"

# Hard-link publication is atomic and fails rather than replacing an artifact
# created concurrently by another process.
if ! ln "$temporary" "$DESTINATION"; then
  verify "$DESTINATION"
fi
echo "Official TR3D checkpoint verified: $DESTINATION"
