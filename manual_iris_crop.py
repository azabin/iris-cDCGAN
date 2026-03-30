import os
import cv2
import numpy as np
import math

# -----------------------------
# CONFIGURATION
# -----------------------------
input_dir = input("Enter input directory path: ").strip()
output_dir = input("Enter output directory path: ").strip()
os.makedirs(output_dir, exist_ok=True)

print("\nInstructions:")
print(" 1️⃣ Left-click on the center of the pupil.")
print(" 2️⃣ Left-click again on the iris boundary (anywhere on the circle).")
print(" The program will crop a circular region around the iris and save it.")
print(" Press [n] to skip to the next image, [q] to quit.\n")

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
click_points = []  # stores two clicks per image


def mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"Clicked at: ({x}, {y})")
        

def crop_iris(image, center, perimeter_point):
    """Crop a circular region based on two click points."""
    cx, cy = center
    px, py = perimeter_point
    radius = int(math.sqrt((px - cx) ** 2 + (py - cy) ** 2))

    # Create a circular mask
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    # Apply the mask
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    # Crop bounding square around circle
    x1, y1 = max(cx - radius, 0), max(cy - radius, 0)
    x2, y2 = min(cx + radius, image.shape[1]), min(cy + radius, image.shape[0])
    cropped = masked_img[y1:y2, x1:x2]
    return cropped


# -----------------------------
# MAIN LOOP
# -----------------------------
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]

for idx, img_name in enumerate(sorted(image_files)):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Skipping unreadable file: {img_name}")
        continue

    click_points = []
    cv2.namedWindow("Select Iris", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Iris", mouse_callback)

    print(f"\n[{idx + 1}/{len(image_files)}] Processing: {img_name}")

    while True:
        disp_img = img.copy()
        for pt in click_points:
            cv2.circle(disp_img, pt, 3, (0, 255, 0), -1)
        cv2.imshow("Select Iris", disp_img)

        key = cv2.waitKey(1) & 0xFF

        if len(click_points) == 2:
            cropped_img = crop_iris(img, click_points[0], click_points[1])
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, cropped_img)
            print(f"✅ Saved cropped image: {out_path}")
            break

        elif key == ord("n"):  # skip image
            print("⏭️ Skipped.")
            break

        elif key == ord("q"):  # quit
            print("👋 Exiting.")
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyAllWindows()

print("\n🎯 Done! All available images processed.")