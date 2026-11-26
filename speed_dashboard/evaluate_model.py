"""Measure a model on your labelled test images and save the numbers for the dashboard's
"Model quality" tab.

    python evaluate_model.py models/best_v1.pt --data C:/path/to/speed_signs/dataset/data_local.yaml
    (use --split valid to measure on the validation set instead)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def count_instances(data_yaml, split):
    """Count signs / images per class straight from the label files of that split."""
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    img_dir = Path(cfg.get("path", Path(data_yaml).parent)) / cfg[split]
    lab_dir = Path(str(img_dir).replace("images", "labels"))
    inst, imgs = defaultdict(int), defaultdict(int)
    for f in lab_dir.glob("*.txt"):
        seen = set()
        for line in f.read_text().splitlines():
            if line.strip():
                c = int(line.split()[0]); inst[c] += 1; seen.add(c)
        for c in seen:
            imgs[c] += 1
    return inst, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "valid"])
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args()
    split = "val" if a.split == "valid" else a.split

    from ultralytics import YOLO
    model = YOLO(a.model)
    r = model.val(data=a.data, split=split, imgsz=a.imgsz, plots=False, verbose=False)
    names = [model.names[i] for i in sorted(model.names)]
    inst, imgs = count_instances(a.data, split)
    classes = []
    for i, cid in enumerate(r.box.ap_class_index):
        p, rc, ap50, ap5095 = r.box.class_result(i)
        classes.append({"name": names[int(cid)], "images": imgs[int(cid)], "instances": inst[int(cid)],
                        "precision": round(float(p), 3), "recall": round(float(rc), 3),
                        "map50": round(float(ap50), 3), "map50_95": round(float(ap5095), 3)})
    out = {"model": Path(a.model).name, "split": "test" if split == "test" else "validation",
           "note": f"Measured on the {a.split} set. Classes with no example in it are NOT evaluated.",
           "overall": {"precision": round(float(r.box.mp), 3), "recall": round(float(r.box.mr), 3),
                       "map50": round(float(r.box.map50), 3), "map50_95": round(float(r.box.map), 3)},
           "class_names": names, "classes": classes}
    dest = Path(__file__).parent / "models" / (Path(a.model).stem + ".metrics.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved", dest)


if __name__ == "__main__":
    main()
