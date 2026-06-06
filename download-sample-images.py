import os
import urllib.request
import sys

def main():
    dest_dir = os.path.join("samples", "images")
    os.makedirs(dest_dir, exist_ok=True)

    images = {
        "bus.jpg": "https://ultralytics.com/images/bus.jpg",
        "zidane.jpg": "https://ultralytics.com/images/zidane.jpg"
    }

    print("Starting download of sample images...")
    for filename, url in images.items():
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            print(f"Already exists: {dest_path}")
            continue

        print(f"Downloading {filename} from {url}...")
        try:
            # Use urllib to fetch the image files
            urllib.request.urlretrieve(url, dest_path)
            print(f"Success: Saved {filename} to {dest_path} ({os.path.getsize(dest_path)} bytes)")
        except Exception as e:
            print(f"Error downloading {filename}: {e}", file=sys.stderr)

    print("\nDownload process completed.")

if __name__ == "__main__":
    main()
