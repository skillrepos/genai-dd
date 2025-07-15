import numpy as np
 
class NeuralNet():

    def __init__(self):
        # Seed the random number generator for reproducibility
        np.random.seed(1)

        # Initialize weights randomly with mean 0
        self.weights = 2 * np.random.random((3, 1)) - 1

    # Sigmoid activation function 
    # takes a number as input and returns a value between 0 and 1
    # determines if a node should be activated or not
    # activated = contributes to the calculations of the net
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Derivative of the sigmoid function
    # used during backpropagation to determine how much to adjust the weights
    # up or down to get the net's predictions closer to the training output
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Our training function
    def train_net(self, training_inputs, training_outputs, training_iterations):

        # Learning rate
        # controls how quickly a neural net "learns"
        # determines how far the neural network weights change within the context of optimization while minimizing the error
        learning_rate = 0.1
    
        # Training iterations
        for iteration in range(training_iterations):
            # Forward propagation (feed-forward)
            outputs = self.process(training_inputs)
    
            # Figure out how much we're off - the error rate for back-propagation
            error = training_outputs - outputs
    
            # Perform the weight adjustments for back-propagation
            adjustments = error * self.sigmoid_derivative(outputs)
            self.weights += np.dot(training_inputs.T, adjustments) * learning_rate


    def process(self, inputs):

        input_layer = inputs.astype(float)  
        outputs = self.sigmoid(np.dot(input_layer, self.weights))
        return outputs

if __name__ == "__main__":


    # training data input has 4 examples (each row is one set)
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])
    
    # training data output has 4 values - each item corresponds to input row
    training_outputs = np.array([[0],
                                 [1],
                                 [0],
                                 [1]])
    
    # initialize the neural net
    neural_net = NeuralNet()
    
    print("Randomly generated starting weights: ")
    print(neural_net.weights)

    training_iterations = 10000

    # do the training
    neural_net.train_net(training_inputs, training_outputs, training_iterations)
    
    # Output the results
    print("Weights After Training:")
    print (neural_net.weights)

    # now we can give the trained model some input values of our own to try
    input_one = str(input("Input one: "))
    input_two = str(input("Input two: "))
    input_three = str(input("Input three: "))
    
    print("Processing new inputs: ", input_one, input_two, input_three)
    print("Predicted results: ")
    print(neural_net.process(np.array([input_one, input_two, input_three])))
