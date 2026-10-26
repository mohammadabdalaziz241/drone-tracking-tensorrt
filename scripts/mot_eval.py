"""Evaluate tracker outputs with TrackEval (HOTA, CLEAR/MOTA, Identity/IDF1).

Example:
  python scripts/mot_eval.py --trackers y640_bytetrack y1280_botsort
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import trackeval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trackers", nargs="+", required=True)
    p.add_argument("--eval-root", default="/scratch/ma06314/drone-tracking/mot_eval")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.eval_root)

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(dict(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False,
                         OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False, OUTPUT_EMPTY_CLASSES=False))
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(dict(GT_FOLDER=str(root / "gt"), TRACKERS_FOLDER=str(root / "trackers"),
                       BENCHMARK="VisDrone", SPLIT_TO_EVAL="val", TRACKERS_TO_EVAL=args.trackers,
                       SEQMAP_FILE=str(root / "gt" / "seqmaps" / "VisDrone-val.txt"),
                       CLASSES_TO_EVAL=["pedestrian"], DO_PREPROC=False, PRINT_CONFIG=False))

    res, _ = trackeval.Evaluator(eval_cfg).evaluate(
        [trackeval.datasets.MotChallenge2DBox(ds_cfg)],
        [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])

    rows = {}
    for name in args.trackers:
        r = res["MotChallenge2DBox"][name]["COMBINED_SEQ"]["pedestrian"]
        run = json.loads((root / "trackers" / "VisDrone-val" / name / "run.json").read_text())
        rows[name] = {
            "HOTA": round(100 * float(r["HOTA"]["HOTA"].mean()), 2),
            "DetA": round(100 * float(r["HOTA"]["DetA"].mean()), 2),
            "AssA": round(100 * float(r["HOTA"]["AssA"].mean()), 2),
            "MOTA": round(100 * float(r["CLEAR"]["MOTA"]), 2),
            "IDF1": round(100 * float(r["Identity"]["IDF1"]), 2),
            "IDSW": int(r["CLEAR"]["IDSW"]),
            "TP": int(r["CLEAR"]["CLR_TP"]),
            "FP": int(r["CLEAR"]["CLR_FP"]),
            "FN": int(r["CLEAR"]["CLR_FN"]),
            "DetPr": round(100 * float(r["HOTA"]["DetPr"].mean()), 2),
            "DetRe": round(100 * float(r["HOTA"]["DetRe"].mean()), 2),
            "FPS": run["fps_end_to_end_python"],
        }

    print("| Run | HOTA | DetA | AssA | MOTA | IDF1 | ID switches | FPS (Python) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for n, m in rows.items():
        print(f"| {n} | {m['HOTA']} | {m['DetA']} | {m['AssA']} | {m['MOTA']} | {m['IDF1']} | {m['IDSW']} | {m['FPS']} |")
    print("\n| Run | TP | FP | FN | Det. precision | Det. recall |")
    print("|---|---:|---:|---:|---:|---:|")
    for n, m in rows.items():
        print(f"| {n} | {m['TP']} | {m['FP']} | {m['FN']} | {m['DetPr']} | {m['DetRe']} |")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
