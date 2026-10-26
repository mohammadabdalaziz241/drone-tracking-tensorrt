"""Convert VisDrone2019-MOT annotations to the MOTChallenge layout that TrackEval expects.

VisDrone-MOT annotation columns:
  frame, target_id, x, y, w, h, score, category, truncation, occlusion
  score: 1 = evaluated, 0 = ignored;  category 0 = "ignored region", 11 = "others"

Following the VisDrone-MOT protocol, only five categories are evaluated (pedestrian, car, van,
truck, bus), class-agnostically. Ignored regions are saved separately so that predictions
falling inside them can be removed before evaluation.

Output:
  <out>/gt/VisDrone-val/<seq>/gt/gt.txt        frame,id,x,y,w,h,1,1,1
  <out>/gt/VisDrone-val/<seq>/seqinfo.ini
  <out>/gt/seqmaps/VisDrone-val.txt
  <out>/ignored_regions.json                    {seq: {frame: [[x, y, w, h], ...]}}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL_CATEGORIES = {1: "pedestrian", 4: "car", 5: "van", 6: "truck", 9: "bus"}
IGNORED_REGION = 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--visdrone", default="/scratch/ma06314/drone-tracking/datasets/VisDrone-MOT/VisDrone2019-MOT-val")
    p.add_argument("--out", default="/scratch/ma06314/drone-tracking/mot_eval")
    args = p.parse_args()

    root = Path(args.visdrone)
    out = Path(args.out)
    gt_split = out / "gt" / "VisDrone-val"
    seqmap = out / "gt" / "seqmaps" / "VisDrone-val.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)

    ignored = {}
    seqs = sorted(d.name for d in (root / "sequences").iterdir() if d.is_dir())
    total_boxes = 0
    for seq in seqs:
        n_frames = len(list((root / "sequences" / seq).glob("*.jpg")))
        ann = np.loadtxt(root / "annotations" / f"{seq}.txt", delimiter=",", dtype=np.int64, ndmin=2)
        frame, tid, x, y, w, h, score, cat = (ann[:, i] for i in range(8))

        keep = (score == 1) & np.isin(cat, list(EVAL_CATEGORIES))
        rows = np.stack([frame, tid, x, y, w, h], axis=1)[keep]
        rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
        d = gt_split / seq / "gt"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "gt.txt", "w") as f:
            for fr, i, bx, by, bw, bh in rows:
                f.write(f"{fr},{i},{bx},{by},{bw},{bh},1,1,1\n")
        (gt_split / seq / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\nseqLength={n_frames}\nimExt=.jpg\n")

        regions = defaultdict(list)
        for fr, bx, by, bw, bh in np.stack([frame, x, y, w, h], axis=1)[cat == IGNORED_REGION]:
            regions[int(fr)].append([int(bx), int(by), int(bw), int(bh)])
        ignored[seq] = regions
        total_boxes += len(rows)
        print(f"{seq}: {n_frames} frames, {len(rows)} evaluated boxes, "
              f"{len(np.unique(rows[:, 1]))} tracks, {int((cat == IGNORED_REGION).sum())} ignored-region boxes")

    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")
    (out / "ignored_regions.json").write_text(json.dumps(ignored))
    print(f"\n{len(seqs)} sequences, {total_boxes} evaluated boxes -> {out}")


if __name__ == "__main__":
    main()
