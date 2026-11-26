"""python train.py --model yolov8s.pt --epochs 3   --name smoke
   python train.py --model yolov8s.pt --epochs 150 --name v8s
"""
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=960)   
    p.add_argument("--batch", type=int, default=8)     
    p.add_argument("--name", default="train")
    a = p.parse_args()

    here = Path(__file__).parent.resolve()
    root = here / "dataset"
    
    cfg = yaml.safe_load(open(root / "data.yaml"))
    cfg.update(path=root.as_posix(), train="train/images", val="valid/images", test="test/images")
    cfg.pop("roboflow", None)

    data_file = root / "data_local.yaml"
    yaml.safe_dump(cfg, open(data_file, "w"), sort_keys=False)  

    model = YOLO(a.model)  
    model.train(
        data=str(data_file), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=0, workers=2, patience=40, name=a.name,
        project=str(here / "runs"), exist_ok=True,
        degrees=5, hsv_v=0.5, fliplr=0.0,   
        close_mosaic=15, seed=0,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=str(data_file), 
        split="test", 
        imgsz=a.imgsz)
    print("Best model saved at:", best)
    print("TEST mAP50 =", round(metrics.box.map50, 3))


if __name__ == "__main__":
    main()