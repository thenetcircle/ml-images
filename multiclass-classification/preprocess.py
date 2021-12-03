import os
import sys
from glob import glob

import cv2
from tqdm import tqdm

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

input_dir = sys.argv[1]
output_dir = sys.argv[2]

for train_or_test in tqdm(glob(f"{input_dir}/*"), desc="train/test"):
    for category_dir in tqdm(glob(f"{train_or_test}/*"), desc="category"):
        for file_path in tqdm(glob(f"{category_dir}/*.jpg"), desc="images"):
            output_path = f"{output_dir}/{file_path.split('/', maxsplit=1)[1]}"
            if os.path.exists(output_path):
                continue

            temp_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
            img_lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2Lab)

            l, a, b = cv2.split(img_lab)
            img_l = clahe.apply(l)
            img_clahe = cv2.merge((img_l, a, b))

            img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2RGB)
            cv2.imwrite(output_path, img_clahe)
