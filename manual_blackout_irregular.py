import cv2
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
input_dir = input("Enter input directory path: ").strip()
output_dir = input("Enter output directory path: ").strip()
os.makedirs(output_dir, exist_ok=True)

print("\nInstructions:")
print(" - Left-click multiple times to outline unwanted region.")
print(" - Press [Enter/Return] when done; it will close the polygon automatically.")
print(" - The selected region will be filled black and saved.")
print(" - Press [n] to skip an image, [q] to quit.\n")

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
points = []
current_image = None
display_image = None
image_name = None

def draw_points(event, x, y, flags, param):
    global points, display_image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(display_image, (x, y), 3, (0, 255, 0), -1)

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

    points = []
    current_image = img.copy()
    display_image = img.copy()
    cv2.namedWindow("Mask Editor", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Mask Editor", draw_points)

    print(f"\n[{idx + 1}/{len(image_files)}] Editing: {img_name}")

    while True:
        cv2.imshow("Mask Editor", display_image)
        key = cv2.waitKey(1) & 0xFF

        # Press Enter to finalize polygon
        if key == 13 and len(points) > 2:  # Enter key
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            inverted_mask = cv2.bitwise_not(mask)
            masked_img = cv2.bitwise_and(img, img, mask=inverted_mask)
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, masked_img)
            print(f"✅ Saved masked image: {out_path}")
            break

        elif key == ord("n"):  # skip
            print("⏭️ Skipped.")
            break

        elif key == ord("q"):  # quit
            print("👋 Exiting.")
            cv2.destroyAllWindows()
            exit(0)

    cv2.destroyAllWindows()

print("\n🎯 Done! All available images processed.")