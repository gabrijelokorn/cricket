from PIL import Image
import os

for folder in ["noise", "courtship"]:
    for filename in os.listdir(f"assets/clips/{folder}"):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = f"assets/clips/{folder}/{filename}"
        img = Image.open(path)
        if img.size != (16, 300):  # PIL size is (width, height)
            print(path, img.size)