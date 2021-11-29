import torch
import torch.nn as nn
import torch.optim as optim
import time
from Data import Data
import matplotlib.pyplot as plt
import numpy as np

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.dropout = nn.Dropout()
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim = -1)
        
        self.fc1 = nn.Linear(40, 120, bias = False) # , bias = False ??
        self.fc2 = nn.Linear(120, 120, bias = False)
        self.fc3 = nn.Linear(120, 2, bias = False)

        self.bn1 = nn.BatchNorm1d(120)
        self.bn2 = nn.BatchNorm1d(120)

    def forward(self, x):
        x = self.dropout(x)

        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.selu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        #x = self.bn2(x)
        x = self.selu(x)

        x = self.dropout(x)

        x = self.fc3(x)
        x = self.softmax(x)
        return x


class Predictor:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.means = torch.zeros((40))
        self.stds = torch.zeros((40))
        self.model = Model().to(device = self.device)
        try:
            self.model.load_state_dict(torch.load('UFC-model.pickle'))
            self.means = torch.load('means.pt')
            self.stds = torch.load('stds.pt')
        except:
            print('No previously trained model found')
            pass

        self.means = self.means.to(self.device)
        self.stds = self.stds.to(self.device)

    # Method for making predictions using the model
    def predict(self, prediction_data): # prediction_data2 ??
        self.model.eval()
        prediction_data = torch.from_numpy(prediction_data).float().to(self.device)
        prediction_data -= self.means
        prediction_data /= self.stds
        prediction = self.model(prediction_data)
        return prediction

    # Method for training the model using a given set of data
    def train(self, training_data):

        self.model = Model().to(device = self.device)

        num_epochs = 15

        start_time = time.time()
        plot_data = np.empty((num_epochs), dtype = float)

        # The data is split into training data and labels later on
        X = training_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]].values
        y = training_data.iloc[:, [41, 42]].values

        X = torch.tensor(X)
        y = torch.tensor(y)

        self.means = torch.mean(X, dim = 0)
        self.stds = torch.std(X, dim = 0)
        
        X -= self.means
        X /= self.stds

        train_data = []
        for i in range(len(X)):
            train_data.append([X[i], y[i]])

        train_set, val_set = torch.utils.data.random_split(train_data, [2500, X.shape[0] - 2500]) # Splits the training data into a train set and a validation set

        #X = torch.from_numpy(X)

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = 200, shuffle = True, num_workers = 4)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = 200, shuffle = False, num_workers = 4)

        params = []
        params += self.model.parameters()

        criterion = nn.CrossEntropyLoss() #nn.MSELoss() #nn.CrossEntropyLoss()
        optimizer = optim.Adam(params, lr = 1e-4, weight_decay = 1e-12)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [32000, 48000], gamma = 0.1)

        # Checks the performance of the model on the test set
        def check_accuracy(dataset):
            num_correct = 0
            num_samples = 0
            self.model.eval()

            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(dataset):
                    data = data.float().to(device = self.device)
                    labels = labels.to(device = self.device)

                    scores = self.model(data)
                    _, predictions = scores.max(1)
                    labels = torch.max(labels, dim = 1)[1]
                    num_correct += (predictions == labels).sum()
                    num_samples += predictions.size(0)
            
            return (num_correct * 100 / num_samples).item()


        for epoch in range(num_epochs):
            print('Epoch: ', epoch)
            train_loss = 0.0
            self.model.train()

            for batch_idx, (data, labels) in enumerate(train_dataloader):                
                data = data.float().to(device = self.device)
                labels = labels.to(device = self.device)

                scores = self.model(data) # Runs a forward pass of the model for all the data
                loss = criterion(scores.float(), labels.float()).float() # Calculates the loss of the forward pass using the loss function
                train_loss += loss

                optimizer.zero_grad() # Resets the optimizer gradients to zero for each batch
                loss.backward() # Backpropagates the network using the loss to calculate the local gradients

                optimizer.step() # Updates the network weights and biases

            valid_loss = 0.0
            self.model.eval()

            for batch_idx, (data, labels) in enumerate(val_dataloader):
                with torch.no_grad():
                    data = data.float().to(device = self.device)
                    labels = labels.to(device = self.device)
                    
                    target = self.model(data)
                    loss = criterion(target, labels).float()
                    valid_loss = loss.item() * data.size(0)

            scheduler.step()

            valid_accuracy = check_accuracy(val_dataloader)
            print(valid_accuracy, '% Validation Accuracy')

            plot_data[epoch] = valid_loss

        print('Finished in %s seconds' % round(time.time() - start_time, 1))
        plt.plot(plot_data)
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.show()

        torch.save(self.model.state_dict(), 'UFC-model.pickle')
        print('Saved model to .pickle file')
        torch.save(self.means, 'means.pt')
        torch.save(self.stds, 'stds.pt')
        print('Saved means and standard deviations')


if __name__ == '__main__':
    predictor = Predictor()
    data = Data()
    training_data = data.training_data
    predictor.train(training_data)