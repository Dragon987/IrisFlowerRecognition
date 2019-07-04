import tensorflow as tf
from keras.models import model_from_json
import numpy as np

class NeuralNet:
    def __init__(self, json_file_name, weights_file):
        with open(json_file_name, 'r') as file:
            model_str = file.read()
        self.model = model_from_json(model_str)
        self.model.load_weights(weights_file)
        self.graph = tf.get_default_graph()
    
    def make_prediction(self, inputs):
        inputs = inputs.reshape([1, 4])
        print(inputs.shape)
        with self.graph.as_default():
            pred = np.argmax(self.model.predict(inputs))
            if pred == 0:
                rv = 'setosa'
            elif pred == 1:
                rv = 'versicolor'
            else:
                rv = 'virginica'
            return "Flower is " + rv



