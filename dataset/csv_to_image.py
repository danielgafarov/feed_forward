import numpy as np
from PIL import Image
arr = np.genfromtxt('mnist_test_10.csv',delimiter=',')
number = 0
for line in arr:
    image = np.array_split(line,28)
    image = np.asarray(image)
    image = Image.fromarray(image)
    image = image.convert('RGB')
    image.save(f"test{number}.jpg")
    number += 1