from __future__ import print_function, division
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as function
import torch.optim as optim
import torchvision
import time
import os
import copy
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix


CONST_LEARNING_RATE = 0.001
CONST_EPOCHES_NUMBER = 6
CONST_MOMENTUM = 0.9
CONST_BATCH_SIZE = 4


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(function.relu(self.conv1(x)))
        x = self.pool(function.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = function.relu(self.fc1(x))
        x = function.relu(self.fc2(x))
        x = self.fc3(x)
        return function.log_softmax(x, dim=1)


def load_data_part_1():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    # Define the indices
    indices = list(range(len(train_set)))  # start with all the indices in training set
    split = int(len(train_set) * 0.2)  # define the split size

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Define our samplers.
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the loaders.
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=CONST_BATCH_SIZE, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=CONST_BATCH_SIZE,
                                                    sampler=validation_sampler)

    return train_loader, validation_loader, test_loader


def train(net, train_loader, validation_loader, resnet, device, train_loss, validation_loss):

    return_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    if resnet == 1:
        parameters = filter(lambda p: p.requires_grad,net.parameters())
    else:
        parameters = net.parameters()

    optimizer = optim.SGD(parameters, lr=CONST_LEARNING_RATE, momentum=CONST_MOMENTUM)

    # loop over the dataset multiple times
    for epoch in range(CONST_EPOCHES_NUMBER):
        running_loss = 0.
        correct = 0.0
        total = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            return_loss += loss.item()
            len(train_loader.dataset)
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                return_loss = running_loss
                running_loss = 0.0
        print('==== End Of Epoch ', epoch + 1, ', Accuracy: ', (correct / total) * 100,' ====')
        train_loss.append(return_loss / 2000)
        validation_loss.append(test(net, validation_loader, 0, device, prediction_list = []))

    print('Finished Training')


def test(net, loader, test_flag, device, prediction_list):
    criterion = nn.CrossEntropyLoss()
    return_loss = 0.0
    correct = 0
    total = 0
    counter = 0.0
    labels_list = []
    with torch.no_grad():
        for data in loader:
            counter += 1
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            if test_flag == 1:
                # Add the prediction to the prediction list.
                for i in range(0, 4):
                    output = predicted[i]
                    true = labels[i]
                    labels_list.append(true.item())
                    prediction_list.append(output.item())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            return_loss += criterion(outputs, labels).item()

    print('========== - TEST - Loss: %.3f, Accuracy: %d %% ==========' % (
            return_loss / counter, 100 * correct / total))

    if test_flag == 1:
        print(confusion_matrix(prediction_list, labels_list))
        with open("test.true", "w+") as true:
            true.write('\n'.join(str(v) for v in labels_list))

    return return_loss / counter


def part1(device):
    # Initialize the net.
    net = Net()
    net = net.to(device)

    # Load the data.
    trainLoader, validationLoader, testLoader = load_data_part_1()

    # Prediction to print to the file.
    prediction_list = []

    # Plot the results.
    trainLoss = []
    validationLoss = []
    epochList = np.arange(1, CONST_EPOCHES_NUMBER + 1)

    # Train the net.
    train(net, trainLoader, validationLoader, 0, device, trainLoss, validationLoss)

    # Set the graph.
    fig, ax = plt.subplots()
    plt.plot(epochList, trainLoss, 'r', color="red", label="Train Loss")
    plt.plot(epochList, validationLoss, 'r', color="blue", label="Validation Loss")

    # Set the legend.
    legend = ax.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')  # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')

    # Draw the graph.
    plt.show()

    # Test
    test(net, testLoader, 1, device, prediction_list)

    with open("test.pred", "w+") as pred:
        pred.write('\n'.join(str(v) for v in prediction_list))


def load_data_part_2():
    # Data augmentation and normalization for training, only normalization for validation.
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms)
    test_loader = data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    # Define the indices
    indices = list(range(len(train_set)))  # start with all the indices in training set
    split = int(len(train_set) * 0.2)  # define the split size

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Define our samplers.
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the loaders.
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=CONST_BATCH_SIZE, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=CONST_BATCH_SIZE,
                                                    sampler=validation_sampler)

    return train_loader, validation_loader, test_loader


def part2(device):
    # Initialize the net.
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 10)
    model_conv = model_conv.to(device)

    # Load the data.
    trainLoader, validationLoader, testLoader = load_data_part_2()

    # Train the net.
    train(model_conv, trainLoader, validationLoader, 1, device, train_loss=[], validation_loss=[])

    # Test
    test(model_conv, testLoader, 1, device, prediction_list = [])


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    part1(device)

    part2(device)


if __name__ == "__main__":
    main()