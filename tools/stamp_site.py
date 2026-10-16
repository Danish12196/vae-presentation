"""List every file the site needs and give the service worker a new cache version.

    python tools/stamp_site.py

Run it after changing anything in site/ (and after training/export.py).
It writes site/precache.json (every file with its size) and rewrites the VERSION line
in site/sw.js with a hash of all the files, so visitors' browsers pick up the change.
"""

import hashlib
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
SKIP_NAMES = {"sw.js", "precache.json", ".gitkeep"}
SKIP_SUFFIXES = {".map", ".txt"}


def site_files():
    for path in sorted(SITE.rglob("*")):
        rel = path.relative_to(SITE).as_posix()
        if (path.is_file() and path.name not in SKIP_NAMES and path.suffix not in SKIP_SUFFIXES
                and not path.name.startswith("LICENSE")):
            yield rel, path


def main():
    digest = hashlib.sha256()
    files = []
    for rel, path in site_files():
        data = path.read_bytes()
        digest.update(rel.encode() + b"\0" + data)
        files.append({"path": rel, "bytes": len(data)})
    version = digest.hexdigest()[:12]
    total = sum(f["bytes"] for f in files)

    (SITE / "precache.json").write_text(
        json.dumps({"version": version, "total_bytes": total, "files": files}, indent=1) + "\n")
    sw = SITE / "sw.js"
    text = sw.read_text(encoding="utf-8")
    text = re.sub(r'^const VERSION = ".*";$', f'const VERSION = "{version}";', text, count=1, flags=re.M)
    sw.write_text(text, encoding="utf-8")
    print(f"{len(files)} files, {total / 1e6:.1f} MB, version {version}")


if __name__ == "__main__":
    main()
