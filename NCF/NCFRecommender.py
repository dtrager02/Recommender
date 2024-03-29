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
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import kaiming_uniform_
import torch.nn as nn
import math
from DataPrep import *

class ExplicitNCF(Module):
    def __init__(self, num_users, num_items, train_dl,test_dl,batch_size=32):
        super(ExplicitNCF,self).__init__()
        hidden_size = 8
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=17, out_features=hidden_size)
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        # self.time_input = nn.Linear(in_features=1, out_features=8)
        # kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size/2)
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.output = nn.Linear(in_features=hidden_size/2, out_features=1)
        self.mlp = []
        dims = [hidden_size,hidden_size,hidden_size/2]
        for i in range(1,3):
            layer = nn.Linear(in_features=dims[i-1], out_features=dims[i])
            kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.mlp.append(layer)
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(rate=.2))
        self.mlp.append(nn.Linear(in_features=dims[-1], out_features=1))
        self.mlp = nn.Sequential(*self.mlp)
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.batch_size = batch_size

    def forward(self, user_input, item_input,time_input):
        
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        # print(user_embedded.size(),item_embedded.size(),time_input.size())
        # Concat the two embedding layers
        time_input = torch.reshape(time_input,(user_input.size()[0],1))
        vector = torch.cat([user_embedded, item_embedded,time_input], dim=1)
        # print(vector.size())
        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = self.output(vector)

        return pred
        
    # train the model
    def fit(self,train_dl,epochs=20,lr=.01):
        # define the optimization
        criterion = MSELoss()
        optimizer = SGD(self.parameters(), lr=lr, momentum=0.9)
        # enumerate epochs
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
                optimizer.zero_grad()
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
                print(np.mean(predictions),np.mean(actuals),predictions[0:4])
                print(f"Train RMSE: {np.sqrt(np.mean((predictions-actuals)**2))}")
            
            
                
             
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

if __name__ == "__main__":
    
    data = np.load("../movielense_27.npy")
    drop_indices = np.random.choice(data.shape[0],size=int(data.shape[0]/10),replace=False)
    test_samples = data[drop_indices,:]
    train_samples = np.delete(data,drop_indices,axis=0)
    # train_dataloader = DataLoader(train_samples, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_samples, batch_size=64, shuffle=True)
    