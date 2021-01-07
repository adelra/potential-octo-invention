import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import stats
import seaborn
import matplotlib.pyplot as plt
train_set = torchvision.datasets.FashionMNIST(root="../data/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.ff1 = nn.Linear(9216, 128)
        self.ff2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.ff1(x)
        x = self.dropout2(x)
        x = self.ff2(x) 
        out = F.softmax(x, dim=1)
        return(out)

model = Net()
epochs =1  
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
loss_func = torch.nn.CrossEntropyLoss()
if not os.path.exists("model.pkl"):
    for epoch in range(epochs):
       for batch_id, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
    print(f"epoch: {epoch}, Loss: {loss}")

print("Saving training finished, saving the model...")
torch.save(model, "model.pkl")

print("evaluating on test dataset...")
test_set = torchvision.datasets.FashionMNIST(root="../data/", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print("finding biases...")
test_set = torch.rand(10,1,28,28) 

# Import MNIST and use MNIST on the model
mnist_test_set = torchvision.datasets.MNIST(root="../data/", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_test_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=10)
examples_mnist = enumerate(mnist_test_loader)
batch_idx, (example_data_mnist, example_targets_mnist) = next(examples_mnist)

model.eval()


with torch.no_grad():
    outputs=model(test_set)
    print(f"Random input: {outputs}")
    outputs_mnist = model(example_data_mnist) 
    print(f"MNIST dataset:{outputs_mnist}")
    outputs_test = model(example_data)
    print(f"Test data:{outputs_test}")
