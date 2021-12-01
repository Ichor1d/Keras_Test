from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np

if __name__ == '__main__':
    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    datapoints = dataset[:, 0:8]
    class_of_datapoint = dataset[:, 8]

    # define the keras model parameters
    input_dim = 8
    size_first_hl = 12
    size_second_hl = 10
    size_output = 1

    # define the keras model
    model = Sequential()
    """
    in your case this SHOULD be an Embedding Layer. But I am not 100% sure if that really is the case.
    """
    model.add(Input(input_dim))
    model.add(Dense(size_first_hl, activation='relu'))
    model.add(Dense(size_second_hl, activation='relu'))

    """
    the sigmoid function maps any input on a scale between 0 and 1.
    The output after the sigmoid function represents the probability of the input to belong to class 1.
    If you only have 2 classes a single neuron is sufficient. Everything after this needs to match the amount of classes
    
    If you want to repeat this for more than two classes you need to:
     a) increase the size of the output layer to the number of classes
     b) use the softmax activation
    """
    model.add(Dense(size_output, activation='sigmoid'))

    """
    The optimizer should not really matter. I would just go with Adam.
    The loss needs to be changed for multiple classes, since the binary cross entropy loss is only valid for..
    well.. binary outputs
    """
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset, consisting of data points
    model.fit(datapoints, class_of_datapoint, epochs=150, batch_size=10)

    # evaluate the model
    """
    Here you would usually use the TEST set. I have skipped this step, since it should just show how the code works.
    """
    _, accuracy = model.evaluate(datapoints, class_of_datapoint)
    print('Accuracy: %.2f' % (accuracy*100))
