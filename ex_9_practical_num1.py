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
import sklearn
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix


CONST_LEARNING_RATE = 0.001
CONST_EPOCHES_NUMBER = 6  # - TODO - change to 5!!
CONST_MOMENTUM = 0.9
CONST_BATCH_SIZE = 4

CONST_TEST_SIZE = 2500
CONST_FILTERS_NUMBER = 2  # TODO - REMOVE?
CONST_NAME = "CIFAR10"
CONST_DEVICE = "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(function.relu(self.conv1(x)))
        x = self.pool(function.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
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


def train_part_1(net, train_loader, validation_loader, train_loss, validation_loss):

    return_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=CONST_LEARNING_RATE, momentum=CONST_MOMENTUM)

    # loop over the dataset multiple times
    for epoch in range(CONST_EPOCHES_NUMBER):
        running_loss = 0.
        correct = 0.0
        total = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

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
        validation_loss.append(test_part_1(net, validation_loader, prediction_list = []))

    print('Finished Training')


def test_part_1(net, loader, prediction_list):
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
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            if (len(loader) == CONST_TEST_SIZE):
                # Add the prediction to the prediction list.
                for i in (0, 3):
                    output = predicted[i]
                    true = labels[i]
                    labels_list.append(true.item())
                    prediction_list.append(output.item())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            return_loss += criterion(outputs, labels).item()

    print('========== - TEST - Loss: %.3f, Accuracy: %d %% ==========' % (
            return_loss / counter, 100 * correct / total))

    if (len(loader) == CONST_TEST_SIZE):
        confusion_matrix(prediction_list, labels_list)
        with open("test.true", "w+") as true:
            true.write('\n'.join(str(v) for v in labels_list))

    return return_loss / counter


def part1():
    # Initialize the net.
    net = Net()

    # Load the data.
    trainLoader, validationLoader, testLoader = load_data_part_1()

    # Prediction to print to the file.
    prediction_list = []

    # Plot the results.
    trainLoss = []
    validationLoss = []
    epochList = np.arange(1, CONST_EPOCHES_NUMBER + 1)

    # Train the net.
    train_part_1(net, trainLoader, validationLoader, trainLoss, validationLoss)

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
    test_part_1(net, testLoader, prediction_list)

    with open("test.pred", "w+") as pred:
        pred.write('\n'.join(str(v) for v in prediction_list))





























































def load_data_part_2():
    # Data augmentation and normalization for training, only normalization for validation.
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),transforms.ToTensor()
                                        ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }

    data_dir = 'hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device(CONST_DEVICE)

    return dataloaders, dataset_sizes


def train_part_2(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(CONST_EPOCHES_NUMBER):
        print('Epoch {}/{}'.format(epoch, CONST_EPOCHES_NUMBER - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(CONST_DEVICE)
                labels = labels.to(CONST_DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def part2():
    # Initialize the net.
    model_ft = models.resnet18(pretrained=True)
    num_filters = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_filters, 2)

    model_ft = model_ft.to(CONST_DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=CONST_LEARNING_RATE, momentum=CONST_MOMENTUM)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Load the data.
    dataloaders, dataset_sizes = load_data_part_2()

    # Train the net.
    model_ft = train_part_2(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes,
                            num_epochs=CONST_EPOCHES_NUMBER)

    # Test
    test_part_1(model_ft, dataloaders['val'])


def main():

    part1()

    #part2()


if __name__ == "__main__":
    main()