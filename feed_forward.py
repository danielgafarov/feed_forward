#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def generate_target(number):
    result = np.full(10,0.01)
    result[number] = 0.99
    return result

class NeuralNetwork():

    #initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.wih = np.empty([hiddennodes,inputnodes])
        for i in range(hiddennodes):
            self.wih[i] = np.random.random(inputnodes) - 1
        self.who = np.empty([outputnodes,hiddennodes])
        for i in range(outputnodes):
            self.who[i] = np.random.random(hiddennodes) - 1
        self.lr = learningrate

    #sigmoid function
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    #derivative of sigmoid function
    def sigmoid_derivative(self, x):
        Sx = self.sigmoid(x)
        return Sx * (1 - Sx)
    
    #train the neural network
    def train(self, inputs, targets,iterations):
        for i in range(iterations):
            for i in range(inputs.shape[0]):
                #think abouts input
                think = self.think(inputs[i])
                hidden = think[0]
                output = think[1]
                #generate target array to compare output with actual value
                target = generate_target(targets[i])
                #calculate errors
                Eout = target - output
                Ehidden = np.dot(self.who.T,Eout)
                #calculate deltas using errors and update weights
                delta_who = np.dot(np.atleast_2d(Eout * self.sigmoid_derivative(output)).T,np.atleast_2d(hidden))
                self.who += self.lr * delta_who
                delta_wih = np.dot(np.atleast_2d(Ehidden * self.sigmoid_derivative(hidden)).T, np.atleast_2d(inputs[i]))
                self.wih += self.lr * delta_wih


    #one calculation step of the network
    def think(self, inputs):
        hiddenlayer = self.sigmoid(np.dot(self.wih,inputs))
        return [hiddenlayer, self.sigmoid(np.dot(self.who,hiddenlayer))]
        
        
if __name__ == "__main__":

    input_nodes = 784 #28*28 pixel
    hidden_nodes = 200 #voodoo magic number
    output_nodes = 10 #numbers from [0:9]

    learning_rate = 0.5 #feel free to play around with

    training_data_file = open("mnist_train_full.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    targets = []
    for data in training_data_list:
        targets.append(int(data[0]))
    targets = np.array(targets)
    for i in range(len(training_data_list)):
        training_data_list[i] = training_data_list[i].split(',')[1:]
    scaler = preprocessing.MinMaxScaler(feature_range=(0.01,0.99))
    training_data_list = scaler.fit_transform(training_data_list)

    test_data_file = open("mnist_test_10.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    test_targets = []
    for data in test_data_list:
        test_targets.append(int(data[0]))
    test_targets = np.array(test_targets)
    for i in range(len(test_data_list)):
        test_data_list[i] = test_data_list[i].split(',')[1:]
    test_data_list = scaler.fit_transform(test_data_list)

    n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
    n.train(training_data_list,targets,2)
    number_of_tests = test_targets.size
    correct_guesses = 0
    for i in range(number_of_tests):
        guess = np.argmax(n.think(test_data_list[i])[1])
        if guess == test_targets[i]:
            correct_guesses += 1
    print(float(correct_guesses)/float(number_of_tests))
    """print("plotting image: ")     
    image_array = test_data_list[4].reshape((28,28))
    plt.imshow(image_array,cmap='Greys', interpolation='None')
    plt.show(block = False)"""
# %%
