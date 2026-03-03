# Feed Forward
[![en](https://img.shields.io/badge/lang-en-green.svg)](https://github.com/danielgafarov/feed_forward)

Ein **Feedforward-Neuronales-Netz** multipliziert Eingabewerte mit Gewichten, um Ausgabewerte zu berechnen. Das Netzwerk in diesem Projekt wird mit **28x28 Graustufenbildern** von handgeschriebenen Ziffern aus dem **MNIST-Datensatz** trainiert.

### Netzwerkstruktur
Das Modell besteht aus drei Schichten:

* **Input Layer:** Um jedes Pixel des Eingabebilds verarbeiten zu können, benötigt die Eingangsschicht **784 Perceptrons**.
* **Hidden Layer:** Diese mittlere Schicht verfügt über **200 Perceptrons**.
* **Output Layer:** Damit jede der zehn möglichen Ziffern repräsentiert werden kann, besitzt die Ausgabeschicht **zehn Perceptrons**.