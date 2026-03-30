from PIL import Image
import os

input_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/check"
output_dir = "../data/visible_cropped"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".jpg", ".png", ".bmp")):
        continue
    img = Image.open(os.path.join(input_dir, fname))
    w, h = img.size
    crop_size = min(w, h) // 2
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = (w + crop_size) // 2
    bottom = (h + crop_size) // 2
    cropped = img.crop((left, top, right, bottom))
    cropped.save(os.path.join(output_dir, fname))