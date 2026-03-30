import cv2
import numpy as np
import os

# -----------------------------
# USER INPUT
# -----------------------------
input_dir = input("Enter input directory path: ").strip()
output_dir = input("Enter output directory path: ").strip()
final_size = int(input("Enter desired output image size (e.g., 128): "))
os.makedirs(output_dir, exist_ok=True)

print("\nInstructions:")
print(" - Left-click multiple times to draw a polygon around the iris region.")
print(" - Press [Enter/Return] to finalize the selection.")
print(" - The program will keep only the selected area, apply a smooth circular mask, resize, and save.")
print(" - Press [n] to skip an image or [q] to quit.\n")

points = []
display_img = None

def draw_points(event, x, y, flags, param):
    """Record mouse clicks and display them."""
    global points, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(display_img, (x, y), 3, (0, 255, 0), -1)

def apply_circular_mask(image):
    """Apply a soft-edged circular mask centered in the image."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(h, w) / 2.1)

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Smooth falloff for circular edges
    mask = np.clip((radius - dist) / 40.0, 0, 1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = mask[..., np.newaxis]  # make 3D
    return (image * mask).astype(np.uint8)

# -----------------------------
# MAIN LOOP
# -----------------------------
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

for idx, img_name in enumerate(sorted(image_files)):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping unreadable file: {img_name}")
        continue

    points = []
    display_img = img.copy()
    cv2.namedWindow("Polygon Mask", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Polygon Mask", draw_points)

    print(f"[{idx+1}/{len(image_files)}] Editing {img_name}")

    while True:
        cv2.imshow("Polygon Mask", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(points) > 2:  # Enter key
            # Create polygon mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            masked_img = cv2.bitwise_and(img, img, mask=mask)

            # Crop to polygon bounding box
            x, y, w, h = cv2.boundingRect(pts)
            cropped = masked_img[y:y+h, x:x+w]

            # Resize to final size
            resized = cv2.resize(cropped, (final_size, final_size))

            # Apply smooth circular mask
            final_img = apply_circular_mask(resized)

            # Save
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, final_img)
            print(f"✅ Saved processed image: {out_path}")
            break

        elif key == ord('n'):
            print("⏭️ Skipped.")
            break

        elif key == ord('q'):
            print("👋 Exiting.")
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyAllWindows()

print("\n🎯 All images processed successfully!")
# /Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible/opacity
# /Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/opacity