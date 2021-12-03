import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
from glob import glob

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))

train_img = "test.jpg"

temp_img = cv2.imread(train_img, cv2.IMREAD_COLOR)
temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
img_lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2Lab)

l, a, b = cv2.split(img_lab)
img_l = clahe.apply(l)
img_clahe = cv2.merge((img_l, a, b))

img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)

f, axarr = plt.subplots(1, 2, figsize=(25, 12))

axarr[0].imshow(temp_img)
axarr[1].imshow(img_clahe)
