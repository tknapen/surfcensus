"""One SHA-256 over a whole pack, and whether two packs are the same bytes.

usage: pack_digest.py PACK [PACK2]

Prints the number of files, their size and one SHA-256 over the relative name and the content
of every file under PACK, in name order. With PACK2 it also names the files that differ and
exits with status 1 when any does. It also checks every file that the manifest lists against
the size and SHA-256 that the manifest gives for it (status 1 when one differs).
"""

import hashlib
import json
import sys
from pathlib import Path


def pack_files(pack):
    """{relative name: SHA-256} of every file under a pack directory."""
    pack = Path(pack)
    names = sorted(p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file())
    return {name: hashlib.sha256((pack / name).read_bytes()).hexdigest() for name in names}


def pack_digest(files):
    h = hashlib.sha256()
    for name, digest in sorted(files.items()):
        h.update(f"{name}\0{digest}\n".encode())
    return h.hexdigest()


def listed(manifest):
    """Every file entry ({'file', 'bytes', 'sha256'}) of a manifest."""
    if isinstance(manifest, dict):
        if {"file", "bytes", "sha256"} <= set(manifest):
            yield manifest
        for value in manifest.values():
            yield from listed(value)
    elif isinstance(manifest, list):
        for value in manifest:
            yield from listed(value)


def check_manifest(pack, files):
    """The names of listed files that are absent or differ from what the manifest says."""
    pack = Path(pack)
    manifest = json.loads((pack / "manifest.json").read_text())
    wrong = []
    for entry in listed(manifest):
        path = pack / entry["file"]
        same = files.get(entry["file"]) == entry["sha256"]
        if not same or path.stat().st_size != entry["bytes"]:
            wrong.append(entry["file"])
    unlisted = set(files) - {e["file"] for e in listed(manifest)} - {"manifest.json"}
    return wrong, sorted(unlisted)


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    status = 0
    sets = []
    for pack in sys.argv[1:]:
        if not (Path(pack) / "manifest.json").is_file():
            sys.exit(f"{pack} holds no manifest.json: not a pack of pack_run.py")
        files = pack_files(pack)
        sets.append(files)
        size = sum((Path(pack) / name).stat().st_size for name in files)
        print(f"{len(files)} files, {size} bytes, sha256 {pack_digest(files)}")
        wrong, unlisted = check_manifest(pack, files)
        if wrong or unlisted:
            status = 1
            print(f"  not as the manifest lists them: {wrong}; not listed: {unlisted}")
    if len(sets) == 2:
        differ = sorted(n for n in set(sets[0]) | set(sets[1]) if sets[0].get(n) != sets[1].get(n))
        if differ:
            status = 1
            print(f"{len(differ)} files differ: {differ[:10]}")
        else:
            print("the two packs are the same bytes")
    sys.exit(status)


if __name__ == "__main__":
    main()
