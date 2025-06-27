#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import pickle

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
    def train(self, inputs,targets):
        best_value = 0
        stagnation = 0
        print(f"Learning rate: {self.lr}")
        while (stagnation < 3):
            for i in range(inputs.shape[0]):
                #think about input
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
                
            correct_guesses = 0
            for i in range(inputs.shape[0]):
                guess = np.argmax(n.think(inputs[i])[1])
                if guess == targets[i]:
                    correct_guesses += 1
            if correct_guesses > best_value:
                best_value = correct_guesses
                stagnation = 0
            else:
                stagnation = stagnation + 1
            
            print(f"{(100 * correct_guesses/inputs.shape[0]):.2f}")


    #one calculation step of the network
    def think(self, inputs):
        hiddenlayer = self.sigmoid(np.dot(self.wih,inputs))
        return [hiddenlayer, self.sigmoid(np.dot(self.who,hiddenlayer))]
        
        
if __name__ == "__main__":

    input_nodes = 784 #28*28 pixel
    hidden_nodes = 200 #voodoo magic number
    output_nodes = 10 #numbers from [0:9]

    learning_rate = 0.001 #feel free to play around with
    
    learning_rates = [0.001]

    training_data_file = open("mnist_train_full.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    targets = []
    for data in training_data_list:
        targets.append(int(data[0]))
    targets = np.array(targets)
    for i in range(len(training_data_list)):
        training_data_list[i] = training_data_list[i].split(',')[1:]
    #scaler = preprocessing.MinMaxScaler(feature_range=(0.01,0.99))
    #training_data_list = scaler.fit_transform(training_data_list)
    training_data_list = np.array(training_data_list,dtype=np.float32)/255


    test_data_file = open("dataset/dataset.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    test_targets = []
    for data in test_data_list:
        test_targets.append(int(data[0]))
    test_targets = np.array(test_targets)
    for i in range(len(test_data_list)):
        test_data_list[i] = test_data_list[i].split(',')[1:]
    #test_data_list = scaler.fit_transform(test_data_list)
    test_data_list = np.array(test_data_list,dtype=np.float32)/255
    
    """
    n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
    n.train(training_data_list,targets)
    nn = open("./nn",'wb')
    pickle.dump(n, nn, protocol=None, fix_imports=True, buffer_callback=None)
    nn.close()"""
    
    nn = open("./nn",'rb')
    n = pickle.load(nn)
    nn.close()
    number_of_tests = test_targets.size
    correct_guesses = 0
    for i in range(number_of_tests):
        probabilities = n.think(test_data_list[i])[1]
        guess = np.argmax(probabilities)
        print(probabilities)
        print(test_targets[i])
        print(guess)
        print()
        if guess == test_targets[i]:
            correct_guesses += 1
    print(f"{100 * float(correct_guesses)/float(number_of_tests):.3f}")
    print()
    print()
    print()
    
"""print("plotting image: ")     
image_array = test_data_list[4].reshape((28,28))
plt.imshow(image_array,cmap='Greys', interpolation='None')
plt.show(block = False)"""
# %%
