import os
import random
import shutil

def split_dataset(input_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            out_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)

            for img in split_files:
                src = os.path.join(cls_path, img)
                dst = os.path.join(out_dir, img)
                shutil.copy(src, dst)

        print(f"✅ {cls}: total={n_total}, train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

    print("\n✅ Dataset successfully split into train, val, and test sets.")

if __name__ == "__main__":
    input_dir = input("Enter path to cropped visible dataset: ").strip()
    output_dir = input("Enter path to save split dataset: ").strip()
    split_dataset(input_dir, output_dir)
