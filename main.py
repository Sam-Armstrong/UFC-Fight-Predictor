"""
Author: Sam Armstrong
Date: 2020-2021

Description: Creates the GUI and allows the user to interact with the data (re-scraping the data or creating
the training data) or the deep learning model (making predictions).
"""

from PyTorchPredictor import Predictor
from Data import Data
from Window import Window
from tkinter import *
from tkinter.ttk import *
import time

data = Data() # Instantiates a Data object, which is used for scraping and interacting with all the fighter, fight, and training data
training_data = data.training_data

predictor = Predictor() # Instantiates a Predictor object which is used for interacting with the neural network model

# Trains the model
# try:
#     predictor.train(training_data)
# except:
#     pass

# Gets the chance of each fighter winning using the predictor (through the window)
def getChances():
    name1 = str(fighter_name1.get())
    name2 = str(fighter_name2.get())
    chance1, chance2 = window.getChances(name1, name2, data, predictor)
    chance_label1['text'] = str(chance1) + '%'
    chance_label2['text'] = str(chance2) + '%'

# Produces the training data
def getTrainingData():
    message_label['text'] = 'Creating the training data...'
    window.getTrainingData(predictor, data)
    message_label['text'] = 'Training data sucessfully created.'
    time.sleep(3)
    message_label['text'] = ''

# Scrapes the fight data from the internet
def scrapeData():
    message_label['text'] = 'Scraping the fight data...'
    window.scrapeData(data)
    message_label['text'] = 'Successfully scraped the data.'
    time.sleep(3)
    message_label['text'] = ''
    # Fix bug where the data does not immediately update after scraping


# Creates the Tkinter window
tk = Tk()
tk.title('Fight Predictor')
tk.geometry("500x300+10+20")

# Creates the progress bar
progress = Progressbar(tk, orient = HORIZONTAL, length = 100, mode = 'determinate')
progress.place(x = 360, y = 240)

# Creates a window class that allows communication between the GUI and APIs
window = Window(tk, progress)

# Creates the GUI labels that assist the user
progress_label = Label(tk, text = 'Progress')
progress_label.place(x = 383, y = 215)
message_label = Label(tk, text = '')
message_label.place(x = 190, y = 242)
fighter_label1 = Label(tk, text = 'Fighter 1')
fighter_label1.place(x = 128, y = 40)
fighter_label2 = Label(tk, text = 'Fighter 2')
fighter_label2.place(x = 328, y = 40)
chance_label1 = Label(tk, text = ('%'))
chance_label1.place(x = 130, y = 117)
chance_label2 = Label(tk, text = ('%'))
chance_label2.place(x = 332, y = 117)

fighter_name1 = StringVar()
fighter_name2 = StringVar()

textfield1 = Entry(tk, textvariable = fighter_name1)
textfield1.place(x = 90, y = 80)
textfield2 = Entry(tk, textvariable = fighter_name2)
textfield2.place(x = 290, y = 80)

# Creates the buttons that allow the user to interact with the data and the neural network model
odds_button = Button(tk, text = 'Calculate Odds', command = getChances)
odds_button.place(x = 208, y = 150)

training_button = Button(tk, text = 'Create Training Data', command = getTrainingData)
training_button.place(x = 30, y = 215)

scrape_button = Button(tk, text = 'Re-Scrape Data', command = scrapeData)
scrape_button.place(x = 30, y = 250)

tk.mainloop()