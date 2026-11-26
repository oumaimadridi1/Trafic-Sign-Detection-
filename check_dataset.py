from pathlib import Path 
import collections, yaml

ROOT = Path(__file__).parent / "dataset"
cfg = yaml.safe_load(open(ROOT / "data.yaml"))

names = cfg["names"]
print(f"{len(names)} classes\n") 

total = collections.Counter()
for split in ["train", "valid", "test"]:
    imgs = list((ROOT / split / "images").glob("*"))
    labs = list((ROOT / split / "labels").glob("*.txt"))

    count, empty, bad = collections.Counter(), 0, []

    for lf in labs:
        rows = []
        lines = lf.read_text().splitlines()
        for l in lines:
            if l.strip():
                row = l.split()
                rows.append(row)

        empty += (len(rows) == 0)

        for r in rows:
            ok = len(r) == 5 and int(r[0]) < len(names) and all(0 <= float(x) <= 1 for x in r[1:])
            if ok: 
                count[int(r[0])] += 1
            else: 
                bad.append(lf.name)

    total.update(count)
    
    print(f"[{split}] images={len(imgs)} labels={len(labs)} boxes={sum(count.values())} "
          f"background(no sign)={empty} BAD={len(bad)}")
    
print("\nBoxes per class (all splits). Aim for 100+ each; below 30 = weak:")

for i, n in enumerate(names):
    flag = "  <-- VERY FEW" if total[i] < 15 else ""
    print(f"  {i:2d} {n:40s} {total[i]:4d}{flag}")