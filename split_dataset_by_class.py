import os
import random
import shutil

def split_dataset(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    # Create subfolders
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # Iterate through class folders
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        # Copy files
        for split_name, split_imgs in splits.items():
            split_dir = os.path.join(base_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(class_dir, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)

        print(f"✅ {cls}: train={n_train}, val={n_val}, test={n_test}")

if __name__ == "__main__":
    # Run once for visible and once for NIR
    for modality in ["/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible", "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/NIR"]:
        print(f"\nSplitting {modality}...")
        split_dataset(modality)