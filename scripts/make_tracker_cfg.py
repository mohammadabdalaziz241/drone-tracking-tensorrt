"""Write a copy of an Ultralytics tracker config with some values overridden.

Example:
  python scripts/make_tracker_cfg.py --base botsort.yaml --out configs/trackers/botsort_t0.4.yaml \
      track_high_thresh=0.4 new_track_thresh=0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ultralytics
import yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("overrides", nargs="*", help="key=value pairs")
    args = p.parse_args()

    base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / args.base
    cfg = yaml.safe_load(base.read_text())
    for kv in args.overrides:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown tracker key: {k} (valid: {sorted(cfg)})")
        cfg[k] = yaml.safe_load(v)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{out}: {cfg}")


if __name__ == "__main__":
    main()
