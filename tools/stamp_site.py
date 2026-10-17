"""List every file the site needs, give the service worker a new cache version, check host limits.

    python tools/stamp_site.py

Run it after changing anything in site/ (training/export.py runs it for you).
It writes site/precache.json (every file with its size) and rewrites the VERSION line
in site/sw.js, so visitors' browsers pick up the change.

The version is a hash of data/files.json, models/manifest.json and the contents of every
other shell file. The models and data files carry their own content hash in their names
(export.py), and files.json lists those names, so hashing files.json covers them without
reading 60 MB on every run.

It also checks the static host's limits (Cloudflare Pages: every file under 25 MiB,
fewer than 20,000 files) and prints the 10 largest files.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
# Not precached: the worker itself, its file list, host config, the 404 page, the link-preview image.
SKIP_NAMES = {"sw.js", "precache.json", ".gitkeep", "_headers", "404.html", "og.jpg"}
SKIP_SUFFIXES = {".map", ".txt"}
HASHED = re.compile(r"^(models|data)/.+\.[0-9a-f]{6}\.[^./]+$")  # content hash already in the name

MIB = 1024 * 1024
FILE_LIMIT = 25 * MIB
COUNT_LIMIT = 20_000


def site_files():
    for path in sorted(SITE.rglob("*")):
        rel = path.relative_to(SITE).as_posix()
        if (path.is_file() and path.name not in SKIP_NAMES and path.suffix not in SKIP_SUFFIXES
                and not path.name.startswith("LICENSE")):
            yield rel, path


def check_limits():
    """Every file that gets deployed, against the host's limits. Returns True if all is well."""
    deployed = [(p.stat().st_size, p.relative_to(SITE).as_posix())
                for p in SITE.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    deployed.sort(reverse=True)
    print("\n10 largest files:")
    for size, rel in deployed[:10]:
        print(f"  {size / MIB:7.2f} MiB  {rel}")
    too_big = [rel for size, rel in deployed if size >= FILE_LIMIT]
    total = sum(size for size, _ in deployed)
    print(f"{len(deployed)} files deployed (limit {COUNT_LIMIT:,}), {total / MIB:.1f} MiB on disk, "
          f"largest {deployed[0][0] / MIB:.2f} MiB (limit 25 MiB)")
    for rel in too_big:
        print(f"  TOO BIG: {rel}")
    return not too_big and len(deployed) < COUNT_LIMIT


def main():
    digest = hashlib.sha256()
    files = []
    for rel, path in site_files():
        size = path.stat().st_size
        if not HASHED.match(rel):
            digest.update(rel.encode() + b"\0" + path.read_bytes())
        files.append({"path": rel, "bytes": size})
    version = digest.hexdigest()[:12]
    total = sum(f["bytes"] for f in files)

    (SITE / "precache.json").write_text(
        json.dumps({"version": version, "total_bytes": total, "files": files}, indent=1) + "\n")
    sw = SITE / "sw.js"
    text = sw.read_text(encoding="utf-8")
    text = re.sub(r'^const VERSION = ".*";$', f'const VERSION = "{version}";', text, count=1, flags=re.M)
    sw.write_text(text, encoding="utf-8")
    print(f"{len(files)} files to cache, {total / 1e6:.1f} MB, version {version}")
    if not check_limits():
        sys.exit("host limits exceeded")


if __name__ == "__main__":
    main()
