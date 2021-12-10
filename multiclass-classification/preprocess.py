import os
import sys
from glob import glob
import argparse

import cv2
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument("-i", "--input", help="Input directory", required=True)
    parser.add_argument("-r", "--replace", help="Replace output files", default=False)
    parser.add_argument("-w", "--workers", help="Number of workers", default=os.cpu_count())

    _args = parser.parse_args()

    if not os.path.exists(_args.input_dir):
        print(f"error: input dir doesn't exist: {_args.input_dir}")
        sys.exit(1)

    if not os.path.exists(_args.output_dir):
        try:
            os.mkdir(_args.output_dir)
        except Exception as e:
            print(f"error: could not create output dir '{_args.output_dir}': {str(e)}")
            sys.exit(1)

    return _args


def process_image(_file_path, _output_path):
    temp_img = cv2.imread(_file_path, cv2.IMREAD_COLOR)
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2Lab)

    l, a, b = cv2.split(img_lab)
    img_l = clahe.apply(l)
    img_clahe = cv2.merge((img_l, a, b))

    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2RGB)
    cv2.imwrite(_output_path, img_clahe)


if __name__ == "__main__":
    args = parse_arguments()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

    for train_or_test in tqdm(glob(f"{args.input_dir}/*"), desc="train/test"):
        for category_dir in tqdm(glob(f"{train_or_test}/*"), desc="category"):
            for file_path in tqdm(glob(f"{category_dir}/*.jpg"), desc="images"):
                output_path = f"{args.output_dir}/{file_path.split('/', maxsplit=1)[1]}"

                if os.path.exists(output_path) and not args.replace:
                    continue

                try:
                    process_image(file_path, output_path)
                except Exception as e:
                    print(f"error: could not process file '{file_path}' because: {str(e)}")
