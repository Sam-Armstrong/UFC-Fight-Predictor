"""
Author: Sam Armstrong
Date: 2020-2021

Description: Class that is responsible for containing and handling interactions with the deep learning model.
Allows the model to be trained and predictions to be made.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

class Predictor:
    def __init__(self):
        pass

    # Method for making predictions using the model
    def predict(self, prediction_data1, prediction_data2):
        prediction1 = self.rgr.predict(self.sc.transform([prediction_data1]))[0]
        prediction2 = self.rgr.predict(self.sc.transform([prediction_data2]))[0]

        # Averages the model predictions with fighter data given both ways round
        prediction = []
        prediction.append((float(prediction1[0]) + float(prediction2[1])) / 2)
        prediction.append((float(prediction1[1]) + float(prediction2[0])) / 2)
        return prediction

    # Method for training the model using a given set of data
    def train(self, training_data):
        # Splits the dataset into data and labels
        X = training_data.iloc[:,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]].values
        y = training_data.iloc[:, [41, 42]].values
        y = y.astype(int)

        # Standardizes the training data
        self.sc = StandardScaler()
        X = self.sc.fit_transform(X)

        # Defines the deep neural network regression model
        self.rgr = MLPRegressor(random_state = 0, solver = 'adam', max_iter = 1000000, hidden_layer_sizes = (40, 40, 40),
                                activation = 'relu', learning_rate = 'adaptive')
        self.rgr.fit(X, y) # Trains the model
        self.rgr.out_activation_ = 'softmax' # Softmax is used as the output activation so the output is the probability of each fighter winning
