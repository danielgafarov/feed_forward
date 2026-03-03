# Feed Forward
[![de](https://img.shields.io/badge/lang-de-red.svg)](https://github.com/danielgafarov/feed_forward/blob/main/README-de.md)

A **feedforward neural network** multiplies input values by weights to calculate output values. The network in this project is trained on **28x28 grayscale images** of handwritten digits from the **MNIST dataset**.

### Network Architecture
The model consists of three distinct layers:

* **Input Layer:** To process every pixel of the input image, this layer requires **784 perceptrons** ($28 \times 28$ pixels).
* **Hidden Layer:** This intermediate layer contains **200 perceptrons** for feature extraction.
* **Output Layer:** To identify the specific digit recognized, this layer uses **10 perceptrons**—one for each possible digit (0–9).