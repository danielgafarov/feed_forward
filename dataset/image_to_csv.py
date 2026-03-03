import cv2
import numpy as np
import csv
def csvWriter(file_name, nparray):
        with open(file_name +'.csv', 'w', newline='') as csvfile:
            i = 0
            for array in nparray:
                writer = csv.writer(csvfile, delimiter=',')
                line = array.tolist()
                line.insert(0,i)
                i = i + 1
                writer.writerow(line)

array = []
for j in range(10):
    image = cv2.imread(f'{j}.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = np.empty(0)

    for sub in gray_image:
        img = np.concatenate((img,sub))
    for i in range(len(img)):
        img[i] = abs(img[i] - 255)
        
    array.append(img)
    
csvWriter("dataset", array)
    
    

