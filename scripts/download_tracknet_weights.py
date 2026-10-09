"""Download and verify the selected tennis TrackNet checkpoint."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/vishnushenoy09/tracknet-v1-tennis/resolve/main/"
    "tracknet_weights.pth"
)
SHA256 = "c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076"
DESTINATION = Path("models/tracknet/tracknet_v1_tennis.pth")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.is_file() and file_sha256(DESTINATION) == SHA256:
        print(f"TrackNet checkpoint already verified: {DESTINATION}")
        return 0

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        print(f"Downloading tennis TrackNet checkpoint to {DESTINATION} ...")
        urllib.request.urlretrieve(URL, temporary_path)
        actual_hash = file_sha256(temporary_path)
        if actual_hash != SHA256:
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch: expected {SHA256}, got {actual_hash}"
            )
        shutil.move(str(temporary_path), DESTINATION)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Verified TrackNet checkpoint: {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
