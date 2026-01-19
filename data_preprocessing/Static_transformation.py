import os
import numpy as np
import cv2
import random
from PIL import Image
from tqdm import tqdm

class GaussianNoise:
    def __init__(self, mean=0.0, std=10.0, fluctuation_range=0):
        self.mean = mean
        self.std = std
        self.fluctuation_range = fluctuation_range

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)

        if self.fluctuation_range > 0:
            std = random.randint(max(1, self.std - self.fluctuation_range), self.std + self.fluctuation_range)
        else:
            std = self.std

        noise = np.random.normal(self.mean, std, img_np.shape).astype(np.float32)
        img_noisy = img_np + noise
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(img_noisy)


class Mosaic:
    def __init__(self, mosaic_level=16):
        self.mosaic_level = mosaic_level

    def __call__(self, img: Image) -> Image:
        img = np.array(img)
        h, w, _ = img.shape

        small = cv2.resize(img, (w // self.mosaic_level, h // self.mosaic_level), interpolation=cv2.INTER_LINEAR)

        mosaic_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return Image.fromarray(mosaic_img)


class LowResolution:
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, img: Image) -> Image:
        w, h = img.size

        new_size = (int(w * self.scale), int(h * self.scale))

        img = img.resize(new_size, Image.BILINEAR)

        img = img.resize((w, h), Image.BILINEAR)
        return img

def process_dataset(source_root, output_base_path):
    Static_transformation = {
        "Gaussian": GaussianNoise(mean=0.0, std=25.0, fluctuation_range=0),
        "Mosaic": Mosaic(mosaic_level=5),
        "LowRes": LowResolution(scale=0.1)
    }

    subsets = ['train_images', 'test_images']


    for subset in subsets:
        subset_path = os.path.join(source_root, subset)

        class_folders = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]

        for class_name in tqdm(class_folders, desc=f"Processing {subset}"):
            class_path = os.path.join(subset_path, class_name)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    continue

                try:

                    original_img = Image.open(img_path).convert('RGB')


                    for aug_name, augmenter in Static_transformation.items():

                        aug_img = augmenter(original_img)

                        save_root = os.path.join(output_base_path, f"Image_set_{aug_name}")
                        save_dir = os.path.join(save_root, subset, class_name)

                        os.makedirs(save_dir, exist_ok=True)

                        save_path = os.path.join(save_dir, img_name)
                        aug_img.save(save_path)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

if __name__ == "__main__":
    SOURCE_ROOT = "data/things-eeg/Image_set_Resize"

    OUTPUT_BASE = os.path.dirname(SOURCE_ROOT)

    if os.path.exists(SOURCE_ROOT):
        process_dataset(SOURCE_ROOT, OUTPUT_BASE)
        print("\nAll processing completed!")
    else:
        print(f"Cannot find source path {SOURCE_ROOT}")