# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fetch the fixed official model files and verify every SHA256 digest."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def download_model(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    base = f"{manifest['endpoint']}/api/v1/models/{manifest['source']}/repo"

    def download(entry):
        path = destination / entry["file"]
        if not path.exists() or path.stat().st_size != entry["size"]:
            url = f"{base}?{urlencode({'Revision': entry['revision'], 'FilePath': entry['file']})}"
            temporary = path.with_suffix(f"{path.suffix}.partial")
            with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
            temporary.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}")
        print(f"Verified {entry['file']} ({entry['size']} bytes)", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, manifest["files"]))
    (destination / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    download_model(args.manifest, args.output)


if __name__ == "__main__":
    main()
