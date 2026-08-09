#!/usr/bin/env bash
#
# Regenerate the hashes in gaussian_golden.json, and — the part that actually matters — re-establish
# them against the PRE-EMG parent commit rather than against this tree twice.
#
# The claim the golden encodes is "`--peak-shape gaussian` reproduces the binary that had no
# `--peak-shape` flag". Reproducing this tree from this tree proves nothing, so the procedure is:
#
#   1. Build the parent commit into its OWN target dir (it shares no artifacts with this tree).
#   2. Run every case with the parent binary and NO shape flag — that binary only has the Gaussian.
#   3. Run every case with this tree's binary and `--peak-shape gaussian`.
#   4. Compare, per artifact, using the SAME hash kinds the test uses:
#        signal carriers        -> whole-file sha256, must be equal
#        answer keys            -> parquet data region (file minus thrift footer), must be equal
#        analysis.tdf           -> `sqlite3 .dump` diff, must differ ONLY by the three Sim* rows
#   5. Paste the head hashes into gaussian_golden.json.
#
# Usage:  tests/golden/regenerate.sh <parent-commit> [workdir]
#
set -euo pipefail

PARENT="${1:?usage: regenerate.sh <parent-commit> [workdir]}"
WORK="${2:-${TMPDIR:-/tmp}/timsim-gaussian-golden-regen}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$REPO/tests/golden/gaussian_golden.json"

jqf() { python3 -c "import json,sys;d=json.load(open('$MANIFEST'));print(eval(sys.argv[1]))" "$1"; }

mkdir -p "$WORK"
echo "== 1. build the parent commit ($PARENT) in an isolated worktree =="
if [ ! -d "$WORK/parent" ]; then
  git -C "$REPO" worktree add "$WORK/parent" "$PARENT"
  # .cargo/config.toml is git-ignored (it points the foundation crates at a sibling checkout), so
  # the worktree does not inherit it and would resolve different crate versions without this.
  mkdir -p "$WORK/parent/.cargo"
  cp "$REPO/.cargo/config.toml" "$WORK/parent/.cargo/config.toml" 2>/dev/null || true
  cp "$REPO/Cargo.lock" "$WORK/parent/Cargo.lock"
fi
(cd "$WORK/parent" && CARGO_TARGET_DIR="$WORK/parent-target" cargo build --release --features tdf,thermo,sciex)

echo "== 2. build this tree =="
(cd "$REPO" && cargo build --release --features tdf,thermo,sciex)

echo "== 3/4. run both and compare =="
python3 - "$MANIFEST" "$WORK" "$WORK/parent-target/release" "$REPO/target/release" <<'PY'
import hashlib, json, os, shutil, sqlite3, subprocess, sys

manifest, work, parent_bin, head_bin = sys.argv[1:5]
m = json.load(open(manifest))
fx = m["fixtures"]
PROV = ("'SimPeakShape'", "'SimEmgK'", "'SimNSigma'")

def sha(p, end=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        left = end if end is not None else os.path.getsize(p)
        while left:
            b = f.read(min(1 << 20, left))
            if not b:
                break
            h.update(b); left -= len(b)
    return h.hexdigest()

def parquet_data_end(p):
    with open(p, "rb") as f:
        f.seek(-8, os.SEEK_END); tail = f.read(8)
        assert tail[4:] == b"PAR1", p
        return os.path.getsize(p) - 8 - int.from_bytes(tail[:4], "little")

def dump(p, drop):
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    out = [l for l in con.iterdump() if not (drop and any(k in l for k in PROV))]
    con.close(); return out

bad = False
for case in m["cases"]:
    name = case["name"]
    missing = [k for k in case["needs"] if not os.path.exists(fx[k])]
    if missing:
        print(f"SKIP {name}: fixtures absent: {missing}"); continue
    for side, bindir, shape_flag in (("parent", parent_bin, False), ("head", head_bin, True)):
        out = os.path.join(work, side, name)
        shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        args = []
        skip_next = False
        for a in case["args"]:
            if skip_next:
                skip_next = False; continue
            if a == "--peak-shape" and not shape_flag:
                skip_next = True; continue          # the parent binary has no such flag
            if a.startswith("{") and a.endswith("}"):
                key = a[1:-1]
                args.append(out if key == "out" else fx[key])
            else:
                args.append(a.replace("{out}", out))
        subprocess.run([os.path.join(bindir, case["bin"])] + args, check=True,
                       stdout=subprocess.DEVNULL)
    for art in case["artifacts"]:
        rel, kind = art["path"], art["hash"]
        pp = os.path.join(work, "parent", name, rel)
        hp = os.path.join(work, "head", name, rel)
        if rel.endswith("analysis.tdf"):
            dp, dh = dump(pp, False), dump(hp, True)
            same = dp == dh
            extra = [l for l in dump(hp, False) if any(k in l for k in PROV)]
            print(f"{'OK  ' if same else 'DIFF'} {name}/{rel}: sqlite dump minus provenance; "
                  f"head adds {len(extra)} row(s)")
            for e in extra:
                print("        ", e.strip())
            print(f"        file_sha256(head) = {sha(hp)}")
            print(f"        dump_sha256_without_provenance = "
                  f"{hashlib.sha256(chr(10).join(dh).encode()).hexdigest()}")
        elif kind == "parquet_data_sha256":
            a, b = sha(pp, parquet_data_end(pp)), sha(hp, parquet_data_end(hp))
            same = a == b
            print(f"{'OK  ' if same else 'DIFF'} {name}/{rel}: parquet data {b}")
            print(f"        file_sha256(head) = {sha(hp)}")
        else:
            a, b = sha(pp), sha(hp)
            same = a == b
            print(f"{'OK  ' if same else 'DIFF'} {name}/{rel}: {b}")
        bad |= not same

sys.exit(1 if bad else 0)
PY

echo
echo "Paste the head hashes above into $MANIFEST."
echo "Clean up with: git -C $REPO worktree remove $WORK/parent && rm -rf $WORK"
