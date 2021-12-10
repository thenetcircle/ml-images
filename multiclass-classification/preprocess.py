import os
import sys
import time
from glob import glob
import argparse
import pathlib
import cv2
from tqdm import tqdm
from multiprocessing import Pool


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument("-i", "--input", help="Input directory", required=True)
    parser.add_argument("-r", "--replace", help="Replace output files", default=False)
    parser.add_argument("-w", "--workers", help="Number of workers", default=os.cpu_count())

    _args = parser.parse_args()

    try:
        n_workers = int(float(_args.workers))
    except TypeError:
        print(f"error: worker argument is not a valid int: {_args.workers}")
        sys.exit(1)

    if n_workers < 1 or n_workers > 128:
        print(f"error: worker amount needs to be in range [1, 128] but was '{n_workers}'")
        sys.exit(1)

    if not os.path.exists(_args.input):
        print(f"error: input dir doesn't exist: {_args.input}")
        sys.exit(1)

    if not os.path.exists(_args.output):
        try:
            pathlib.Path(_args.output).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"error: could not create output dir '{_args.output}': {str(e)}")
            sys.exit(1)

    return _args


def process_image(input_tuple):
    try:
        _file_path, _output_path = input_tuple
        temp_img = cv2.imread(_file_path, cv2.IMREAD_COLOR)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2Lab)

        l, a, b = cv2.split(img_lab)
        img_l = clahe.apply(l)
        img_clahe = cv2.merge((img_l, a, b))

        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2RGB)
        cv2.imwrite(_output_path, img_clahe)
        sys.exit(1)
    except Exception as e1:
        print(f"error: could not process file '{file_path}' because: {str(e1)}")


if __name__ == "__main__":
    args = parse_arguments()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    files = list()

    for train_or_test in glob(f"{args.input}/*"):
        for category_dir in glob(f"{train_or_test}/*"):
            output_dir = f"{args.output}/{category_dir.split('/', maxsplit=1)[1]}"

            if not os.path.exists(output_dir):
                try:
                    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"error: could not create output dir '{output_dir}': {str(e)}")
                    sys.exit(1)

            for file_path in glob(f"{category_dir}/*.jpg"):
                output_path = f"{args.output}/{file_path.split('/', maxsplit=1)[1]}"

                if os.path.exists(output_path) and not args.replace:
                    continue

                files.append((file_path, output_path))

    print(len(files))

    with Pool(processes=int(float(args.workers))) as pool:
        results = tqdm(pool.imap(process_image, files), total=len(files))
        tuple(results)  # fetch the lazy results
