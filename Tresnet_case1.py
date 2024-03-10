import os
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.model import ResNet, BasicBlock

# Setup environment variable and hyperparameters
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
batch_size = 32
num_epochs = 100

# Initialize storage for accuracy measurements and predictions
train_acc = np.zeros(num_epochs)
test_acc = np.zeros(num_epochs)
# Adjust the dimensions of test_label_matrix according to your needs
test_label_matrix = np.zeros((27, 1))  # Example: Adjust as necessary

for k in range(27):
    snr = str(k)
    # Initialize model
    net = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=6)

    # Load your data
    data_path = f'./data/case1withnoisevali/snr={snr}ttr=0.5.mat'
    data = sio.loadmat(data_path)

    # Process and load data as tensors
    train_data = torch.from_numpy(data['train_data']).type(torch.FloatTensor).unsqueeze(1)
    train_label = torch.from_numpy(data['train_label'].squeeze()).type(torch.LongTensor) - 1
    # Similar processing for test and validation data...

    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Similar setup for test_loader and vali_loader...

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 250, 300], gamma=0.5)

    # Training and evaluation loop
    for epoch in range(num_epochs):
        net.train()
        for samples, labels in train_loader:
            samples, labels = Variable(samples), Variable(labels)
            optimizer.zero_grad()
            outputs = net(samples)
            # Assuming the first element of the tuple is the logits tensor you want
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Add code for calculating and printing accuracy here...

    # Save the results outside the epoch loop
    # Add code for saving the performance and predictions here...



    # Training accuracy evaluation
    net.eval()
    correct_train, total_train = 0, 0
    for samples, labels in train_loader:
        samples, labels = Variable(samples), Variable(labels)
    outputs = net(samples)
    _, predicted = torch.max(outputs.data, 1)
    total_train += labels.size(0)
    correct_train += (predicted == labels).sum().item()
    train_acc = 100 * correct_train / total_train

    # Test accuracy evaluation
    correct_test, total_test = 0, 0
    predictions = []
    for samples, labels in test_loader:
        samples, labels = Variable(samples), Variable(labels)
    outputs = net(samples)
    _, predicted = torch.max(outputs.data, 1)
    predictions.extend(predicted.cpu().numpy())
    total_test += labels.size(0)
    correct_test += (predicted == labels).sum().item()
    test_acc = 100 * correct_test / total_test

    # Validation accuracy evaluation (if applicable)
    # Ensure to add similar logic for validation set evaluation if you're using it

    # After completing an epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_acc}%, Test Accuracy: {test_acc}%")

    # After all epochs for a SNR level
    # Convert predictions list to a numpy array for saving
    predictions_array = np.array(predictions).reshape(-1, 1)  # Adjust reshape as necessary
    test_label_matrix[k] = predictions_array

    # Saving results
    sio.savemat(f'./results/case1/CM_WithValiCNN_SNR{snr}accuracy.mat', {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'vali_accuracy': vali_acc,  # Include this if you have validation data
        'test_label_matrix': test_label_matrix
    })
