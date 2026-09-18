"""
One-time: fetch YOLO11n and export it to ONNX.

Inference runs on the onnxruntime-gpu this project already uses, so torch and
ultralytics are build-time only — uninstall them afterwards if you want the
environment lean. Keeping one CUDA runtime instead of two also avoids torch and
onnxruntime each reserving their own VRAM pool.

    .venv/bin/python tools/export_person_model.py
"""
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
TARGET = MODELS_DIR / "yolo11n.onnx"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"already present: {TARGET}")
        return

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")          # downloads on first run
    exported = model.export(format="onnx", imgsz=640, simplify=True, opset=12)
    shutil.copy(exported, TARGET)
    print(f"EXPORTED: {TARGET}  ({TARGET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
