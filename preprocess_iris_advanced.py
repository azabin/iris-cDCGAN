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
print(" - Left-click multiple times to draw polygon around the iris region.")
print(" - Press [Enter/Return] to finalize.")
print(" - The program will process, normalize, mask, and save automatically.")
print(" - Press [n] to skip, [q] to quit.\n")

points = []
current_img = None
display_img = None

def draw_points(event, x, y, flags, param):
    global points, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(display_img, (x, y), 3, (0, 255, 0), -1)

def apply_clahe(image):
    """Apply CLAHE on the luminance channel (YCrCb color space)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def color_histogram_normalization(image):
    """Normalize each color channel histogram."""
    result = image.copy()
    for i in range(3):
        result[..., i] = cv2.equalizeHist(result[..., i])
    return result

def apply_circular_mask(image):
    """Apply smooth circular mask based on center and radius."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(h, w) / 2.1)

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = np.clip((radius - dist_from_center) / 40.0, 0, 1)  # 40 controls softness
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = mask[..., np.newaxis]

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
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            masked_img = cv2.bitwise_and(img, img, mask=mask)

            # Crop bounding box of the polygon
            x, y, w, h = cv2.boundingRect(pts)
            cropped = masked_img[y:y+h, x:x+w]

            # Resize to final size
            resized = cv2.resize(cropped, (final_size, final_size))

            # Apply circular soft mask
            blended = apply_circular_mask(resized)

            # Normalize lighting
            clahe_img = apply_clahe(blended)

            # Normalize colors
            normalized = color_histogram_normalization(clahe_img)

            # Save final result
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, normalized)
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