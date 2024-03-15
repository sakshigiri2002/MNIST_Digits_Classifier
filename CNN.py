import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

num_epochs = 5
num_classes = 10
batch_size = 100
#learning_rate = 0.001
learning_rate = 0.0075

DATA_PATH = 'C:\\Users\pooja\PycharmProjects\MNISTData'
MODEL_STORE_PATH = 'C:\\Users\pooja\PycharmProjects\pytorch_models\\'

trans = transforms.Compose([transforms.ToTensor()])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True,)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans,download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(train_dataset)
print(train_loader)

train_X = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_y = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            #nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            #nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4 * 4 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images
        #print('i',i)
        #print('image',images)
        #print('labels',labels)
        # forward pass
        outputs = model(images)
        #print('output',outputs.shape)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Train Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')


fig,ax = plt.subplots()
x1 = len(loss_list)
a1 = np.arange(x1)
x2 = len(acc_list)
a2 = np.arange(x2)
ax.plot(a1,loss_list,label = 'Loss',color = 'red')
ax.plot(a2,acc_list,label = 'accuracy',color = 'blue')
ax.set_xlabel('Batches')
ax.legend()
plt.show()


# lr = 0.0075
#Test Accuracy of the model on the 10000 test images: 97.02 %
#Train Accuracy of the model on the 10000 test images: 97.115 %

# lr = 0.001
#Test Accuracy of the model on the 10000 test images: 98.91 %
#Train Accuracy of the model on the 10000 test images: 99.43 %