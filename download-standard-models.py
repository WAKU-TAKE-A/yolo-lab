import os
import shutil
import sys
from ultralytics import YOLO

def main():
    dest_dir = os.path.join("models", "standard")
    os.makedirs(dest_dir, exist_ok=True)

    models_to_download = [
        "yolov8s.pt",
        "yolov8s-seg.pt",
        "yolo26s.pt",
        "yolo26s-seg.pt"
    ]

    print("Starting download of standard YOLO models...")
    for model_name in models_to_download:
        dest_path = os.path.join(dest_dir, model_name)
        if os.path.exists(dest_path):
            print(f"Already exists: {dest_path}")
            continue

        print(f"\n--- Loading/Downloading {model_name} ---")
        try:
            # Running YOLO(model_name) will download the model file to the CWD
            # if it's not already cached.
            model = YOLO(model_name)
            
            # Check if the file was downloaded to CWD
            if os.path.exists(model_name):
                shutil.move(model_name, dest_path)
                print(f"Success: Downloaded and moved {model_name} to {dest_path}")
            elif os.path.exists(dest_path):
                print(f"Success: {model_name} already located at {dest_path}")
            else:
                # Fallback: check if it was downloaded elsewhere or if it's already in the cache
                # In some versions, it might download to the default cache directory (like ~/.config/Ultralytics/weights)
                # and YOLO(model_name) resolves it. Let's try to save the model to dest_path.
                # Or we can check if we can save it. Wait, model.save() or exporting?
                # Actually, YOLO models can be saved by copying model.ckpt or from model.pt
                # But typically YOLO downloads to the current directory on import.
                # Let's print a warning if we can't find the file in CWD.
                print(f"Warning: Model loaded but {model_name} not found in CWD to move.")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}", file=sys.stderr)

    print("\nDownload process completed.")
    print("Contents of models/standard/:")
    if os.path.exists(dest_dir):
        for f in os.listdir(dest_dir):
            p = os.path.join(dest_dir, f)
            print(f"  - {f} ({os.path.getsize(p)} bytes)")
    else:
        print("  (Directory does not exist)")

if __name__ == "__main__":
    main()
