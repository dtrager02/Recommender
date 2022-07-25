#sbatch -N1 -n1 --gpus=1 --mem-per-gpu=8192 --ntasks=1 --cpus-per-task=16  --constraint=g start.sub
#sacct  --format="JobID,Elapsed,CPUTime,MaxRSS,AveRSS"
#tail -f slurm-146258.out

"""
Notes:
The final model should incorporate a hybrid of MF output and content-user matching
The proportions of these two metrics is determined by how many items the user has rated
the content user matching system will include:
1. Genres of items vs. user genres
2. release dates of items vs typical "era" of user
3. popularity of user-rated items (how niche the user is)
"""

import torch
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
from DataPrep import *

class ExplicitMLP(Module):
    def __init__(self, num_users, num_items, train_dl,test_dl):
        super(ExplicitMLP,self).__init__()
        hidden_size = 8
        self.fc1 = nn.Linear(in_features=num_users+2*num_items, out_features=hidden_size)
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(in_features = hidden_size,out_features=hidden_size)
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.output = nn.Linear(in_features=hidden_size, out_features=1)
        self.dropout = nn.Dropout(.2)
        self.train_dl = train_dl
        self.test_dl = test_dl

    def forward(self, user_input, item_input,time_input):

        vector = torch.cat([user_input, item_input,time_input], dim=1)
        # print(vector.size())
        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        # vector = self.dropout(vector)
        vector = nn.ReLU()(self.fc2(vector))
        # vector = self.dropout(vector)
        # Output layer
        pred = self.output(vector)

        return pred
        
    # train the model
    def fit(self,train_dl,epochs=20,lr=.01):
        # define the optimization
        criterion = MSELoss()
        optimizer = SGD(self.parameters(), lr=lr, momentum=0.9)
        # enumerate epochs
        for i in next(iter(train_dl)):
            print(i.size())
        for epoch in range(epochs):
            #store errors
            current = 0
            predictions,actuals = np.empty(0),np.empty(0)
            # enumerate mini batches
            for inputs in iter(train_dl): #inputs is len() -> 4
                # clear the gradients
                # print("input size",len(inputs))
                user_input, item_input,time_input, labels = inputs # each is size [32]
                # print("divided size",user_input.size())
                optimizer.zero_grad(set_to_none=True)
                # compute the model output
                yhat = self.forward(user_input, item_input,time_input)
                # print(yhat[0])
                # calculate loss
                loss = criterion(torch.reshape(yhat,(-1,)), labels)
                # print(loss)
                with torch.no_grad():
                    predictions = np.hstack((predictions,torch.reshape(yhat,(-1,))))
                    actuals = np.hstack((actuals,torch.reshape(labels,(-1,))))
                    current += len(yhat)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
            with torch.no_grad():
                # print(yhat.size(),labels.size())
                # print(predictions.shape,actuals.shape)
                print(f"Predictions: N({np.mean(predictions)},{np.std(predictions)})")
                print(f"Labels: N({np.mean(actuals)},{np.std(actuals)})")
                print(f"Train RMSE: {np.sqrt(np.mean(np.square(predictions-actuals)))}")
            
            
                
             
    def evaluate(self,test_dl):
        predictions, actuals = np.empty(len(test_dl)), np.empty(len(test_dl))
        current = 0 
        criterion = MSELoss()
        for inputs in iter(test_dl):
            user_input, item_input,time_input, labels = inputs
            # evaluate the model on the test set
            yhat = self(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = labels.numpy()
            actual = actual.reshape((len(actual), 1))
            # store
            predictions[current:current+len(yhat)] = yhat
            actuals[current:current+len(labels)] = labels
            current+= len(yhat)

        with torch.no_grad():
            a = criterion(predictions,actuals)
            print(f"Test loss: {a}")
            return a
