import cv2
import numpy as np
import csv
image = cv2.imread('9.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = np.empty(0)

for sub in gray_image:
    img = np.concatenate((img,sub))
for i in range(len(img)):
    img[i] = abs(img[i] - 255)

def csvWriter(fil_name, nparray):
    example = nparray.tolist()
    with open(fil_name+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(example)

csvWriter("9", img)