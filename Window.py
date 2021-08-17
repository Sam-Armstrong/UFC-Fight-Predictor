"""
Author: Sam Armstrong
Date: 2020-2021

Description: Provides an interface between the GUI and Data/Predictor classes that allow these classes to update the 
progress bar on the interface, while the GUI can acces services such as making predictions or creating the training data.
"""

from tkinter import *
import datetime, pandas

class Window:
    def __init__(self, window, progress):
        self.window = window
        self.progress = progress

    # Uses the trained model to predict the probability of each of two given fighters winning a fight against each other
    def getChances(self, name1, name2, data, predictor):
        date = str(datetime.date.today())

        try:
            prediction_data1 = (data.findFighterStats(name1, date)) + (data.findFighterStats(name2, date))
            prediction_data2 = (data.findFighterStats(name2, date)) + (data.findFighterStats(name1, date))
            prediction = predictor.predict(prediction_data1, prediction_data2)
            chance1 = str(round(prediction[0] * 100, 2))
            chance2 = str(round(prediction[1] * 100, 2))

            return chance1, chance2

        except:
            print('Prediction not possible.')

    # Creates the training data from the known data and re-trains the neural network
    def getTrainingData(self, predictor, data):
        data.createTrainingData(self)

        global training_data
        try:
            training_data = pandas.read_csv('TrainingData.csv')
            predictor.train(training_data)
        except Exception as e:
            print('The training data could not be created. ')

    # Scrapes all the past fights and fighter data from the internet
    def scrapeData(self, data):
        data.getData(self)
        data.getFighterData(self)

    # Updates the progress bar on the window
    def updateProgress(self, n):
        self.progress['value'] = n
        self.window.update_idletasks()
        pass
