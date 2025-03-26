'''
Author: Tom Mathew
Date: 26 March, 2025
'''
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

'''
Using built in library from torchvision,
-we only consider 1 input channel, as we consider images as greyscale
-we can convert each image to a tensor, then normalize each image
'''


torch.manual_seed(42) #randomize weights etc

transform = transforms.Compose([ 
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),          # converting to tensor
    transforms.Normalize((0.5,), (0.5,))  # normalizing (for grayscale), to [-1, 1] range, mean = 0.5, standard deviation = 0.5
])

def load_image(path):
    return transform(Image.open(path))



train_dataset = datasets.ImageFolder(root="./Fashion MNIST Archive/train", transform=transform)
test_dataset = datasets.ImageFolder(root="./Fashion MNIST Archive/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class NN(nn.Module): #creating custom class by inheriting built in class Module
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # fully connected layer with 128 neurons, with input size of 28*28, 
        self.relu = nn.ReLU() #ReLU activation layer, zero if negative, introduce non linearity allows for deep learning
        self.fc2 = nn.Linear(128, 10)  # output layer, to out to 10 classes[0-9]
    

    def forward(self, X):
        
        X = torch.flatten(X,start_dim=1)  #flatten the image or unmatrix it, to get last dimension as 28*28., we start from secondn dimension after batch
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        return X

model = NN()

loss_c = nn.CrossEntropyLoss() 
'''
now we are training the Artificial neural network, with
-built inAdam optimizer
-loss criterion as cross entropy loss, for categorical data, which is our situation

'''
optimizer = optim.Adam(model.parameters(), lr=0.01)
no_epochs = 1
for epoch in range(no_epochs):
    for X_train, y_train in train_loader:
        
        optimizer.zero_grad() #it resets the gradients of all model paramters before backpropagation
        '''
        in pytorch gradients are accumuluated by default, not overwritten, so this step is necessary, to prevent new gradients to be added
        to old gradients
        '''
        outputs = model(X_train)
        loss = loss_c(outputs,y_train)
        loss.backward()
        '''
        backward calls backpropagation, which is crucial in neural networks, to compute gradients wrt to paramters.
        '''
        optimizer.step() #updates model paramters, using computed gradients
    print(f'Epoch: {epoch+1}, loss: {loss.item():.4f}')

def predict(model, X):
    output = model(X)
    _, predicted = torch.max(output, 1)
    return predicted

def evaluate(model, test_loader):
    correct = 0
    total = 0
    
    for X_test, y_test in test_loader:
        predicted = predict(model, X_test)
        
        total += y_test.size(0) #getting number of elements in each batch of y, and adding to total
        
        correct += (predicted == y_test).sum().item() #get scalar value of total correct predictions
    print(f"Test Accuracy: {100 * correct / total:.2f}%") #fix to 2 decimal places

#perform accuracy tests on test data
evaluate(model, test_loader)

#predict an image
print(predict(model, load_image('./Fashion MNIST Archive/train/0/1.png')).item())
