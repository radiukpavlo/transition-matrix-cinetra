"""Verify the protocol registration of CINETRA (CMR-TRANS-AUDIT 6.0.0 and its 6.0.1 amendment).

The registration consists of two signed, annotated git tags in the authors' development repository. This folder
publishes what is needed to check them without that repository (protocol/registration/):

1. Each tag object hashes to its recorded id, and its SSH signature verifies against ``allowed_signers``.
2. Each tag points to a commit, and the commit and tree objects hash to their ids, so they cannot be altered.
3. Walking those trees, each registered path resolves to the recorded blob id (PROOFS.json).
4. Each file published verbatim in this repository has exactly that blob id, so it is byte-identical to the
   registered file. Public editions and hash-only files are reported with their proven registered blob ids.

The dates come from the signer's own clock (the tagger line); they are not third-party timestamps.

Requirements: Python 3.9+, OpenSSH ssh-keygen (8.0 or later) for step 1, and git for step 4 on a clone.
Run from the repository root: python tools/verify_registration.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "protocol" / "registration"


def oid(kind: str, data: bytes) -> str:
    return hashlib.sha1(f"{kind} {len(data)}\0".encode() + data).hexdigest()


def tree_entries(data: bytes) -> dict[str, str]:
    """name -> object id of a raw git tree object."""

    out, pos = {}, 0
    while pos < len(data):
        nul = data.index(b"\0", pos)
        name = data[data.index(b" ", pos) + 1:nul].decode("utf-8")
        out[name] = data[nul + 1:nul + 21].hex()
        pos = nul + 21
    return out


def file_blob(path: Path) -> str:
    """The blob id git assigns to a working-tree file (git applies the repository's .gitattributes)."""

    if shutil.which("git") and (ROOT / ".git").exists():
        proc = subprocess.run(["git", "hash-object", "--", str(path.relative_to(ROOT))], cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout.strip()
    return oid("blob", path.read_bytes())


def verify_signature(tag: str, text: str) -> str:
    keygen = shutil.which("ssh-keygen")
    if not keygen:
        return "SKIPPED (ssh-keygen not found)"
    payload, sig = text.split("-----BEGIN SSH SIGNATURE-----", 1)
    signer = re.search(r"^tagger .*<([^>]+)>", text, re.M).group(1)
    with tempfile.TemporaryDirectory() as tmp:
        sig_file = Path(tmp) / "tag.sig"
        sig_file.write_text("-----BEGIN SSH SIGNATURE-----" + sig, encoding="utf-8", newline="\n")
        proc = subprocess.run([keygen, "-Y", "verify", "-f", str(REG / "allowed_signers"), "-I", signer, "-n", "git",
                               "-s", str(sig_file)], input=payload.encode("utf-8"), capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"FAIL {tag}: signature does not verify\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout.decode(errors="replace").strip() or "Good signature"


def main() -> int:
    proofs = json.loads((REG / "PROOFS.json").read_text(encoding="utf-8"))
    objects: dict[str, tuple[str, bytes]] = {}
    for f in sorted((REG / "objects").iterdir()):
        name, kind = f.name.split(".", 1)
        data = f.read_bytes()
        if oid(kind, data) != name:
            raise SystemExit(f"FAIL object {f.name}: content does not hash to its name")
        objects[name] = (kind, data)
    print(f"objects: {len(objects)} commit and tree objects hash to their ids")

    for tag, rec in proofs["tags"].items():
        data = (REG / "tags" / f"{tag}.tag").read_bytes()
        if oid("tag", data) != rec["tag_object"]:
            raise SystemExit(f"FAIL {tag}: the tag object does not hash to {rec['tag_object']}")
        text = data.decode("utf-8")
        commit = re.search(r"^object ([0-9a-f]{40})$", text, re.M).group(1)
        tree = re.search(rb"^tree ([0-9a-f]{40})$", objects[commit][1], re.M).group(1).decode()
        if commit != rec["commit"] or tree != rec["root_tree"]:
            raise SystemExit(f"FAIL {tag}: the tag, commit and tree do not chain")
        tz = dt.timezone(dt.timedelta(hours=int(rec["tagger_tz"][:3]), minutes=int(rec["tagger_tz"][0] + rec["tagger_tz"][3:])))
        when = dt.datetime.fromtimestamp(rec["tagger_epoch"], tz).strftime("%Y-%m-%d %H:%M:%S %z")
        print(f"{tag}: tag {rec['tag_object'][:12]} -> commit {commit[:12]}, signed {when} (signer's clock); "
              f"signature: {verify_signature(tag, text)}")

    counts: dict[str, int] = {}
    for rec in proofs["files"]:
        node = proofs["tags"][rec["tag"]]["root_tree"]
        for part in rec["path"].split("/"):
            kind, data = objects[node]
            node = tree_entries(data).get(part)
            if node is None:
                raise SystemExit(f"FAIL {rec['tag']}:{rec['path']}: not in the registered tree")
        if node != rec["blob"]:
            raise SystemExit(f"FAIL {rec['tag']}:{rec['path']}: registered blob is {node}, not {rec['blob']}")
        published = rec["published"]
        if published.startswith("verbatim"):
            here = file_blob(ROOT / rec["path"])
            if here != rec["blob"]:
                raise SystemExit(f"FAIL {rec['path']}: the published file (blob {here}) differs from the registered "
                                 f"blob {rec['blob']}")
            published = "verbatim, identical"
        key = f"{rec['tag']} {published}"
        counts[key] = counts.get(key, 0) + 1
    for key, n in sorted(counts.items()):
        print(f"  {n:4d} files: {key}")
    print("PASS: the registration verifies; verbatim files are byte-identical to the registered commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
