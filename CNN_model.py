import torch  # Main torch import for torch tensors
import torch.nn as nn  # Neural network module for building deep learning models
import torch.nn.functional as F  # Functional module, includes activation functions
import torch.optim as optim  # Optimization module
from torchvision import transforms, datasets  # Vision / image processing package built on top of torch
import math

from matplotlib import pyplot as plt  # Plotting and visualization
from sklearn.metrics import accuracy_score  # Computing accuracy metric

DEVICE = "mps"
DIR = ""

# Defined constants created through testing with optuna to find optimzed values.
BATCHSIZE = 5
DROPOUT_P = 0.0840742394198019
'''Percentage of dropout for the dropout layer'''
LEARNING_RATE = 0.0006381447313977647
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.00012098564799135106
'''The weight provided for the L2 regularization'''

def get_mnist():
    '''Gets the MNIST data set and returns the training and testing DataLoader.'''
    
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader

def get_output_size(input_size, padding, stride, kernel):  
    '''
    Calculate the output size of a convolutional or pooling layer.
    
    :param input_size: int, the size of the input feature map 
    :param padding: int, the number of padding pixels added to each side of the input
    :param stride: int, the stride (step size) of the convolution or pooling operation
    :param kernel: int, the size of the kernel (filter) applied during the operation

    :return: int, the size of the output feature map 
    '''
    return math.floor((input_size + 2*padding - kernel)/stride) + 1

class ConvNet(nn.Module):
    def __init__(self):
        '''Initalize the Convolutional Network layers.'''
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding='same')
        self.conv2 = nn.Conv2d(32, 128, kernel_size=(3, 3), padding='same') 
        
        # Pool layers
        self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        
        out1 = get_output_size(input_size=28, padding=1, stride=2, kernel=2)
        out2 = get_output_size(input_size=out1, padding=1, stride=2, kernel=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(DROPOUT_P)

        # Output layers
        self.output_layer = nn.Linear(128*out2*out2, 10) 
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass implementation for the network
        
        :param x: torch.Tensor of shape (batch, 1, 28, 28), input images

        :returns: torch.Tensor of shape (batch, 10), output logits
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.bn2(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1) 
        x = self.output_layer(x)
        return x
    
def train(model, train_loader, loss_fn, optimizer, epoch=-1):
    """
    Trains a model for one epoch (one pass through the entire training data).

    :param model: PyTorch model
    :param train_loader: PyTorch Dataloader for training data
    :param loss_fn: PyTorch loss function
    :param optimizer: PyTorch optimizer, initialized with model parameters
    :kwarg epoch: Integer epoch to use when printing loss and accuracy
    :returns: Accuracy score
    """
    total_loss = 0
    all_predictions = []
    all_targets = []
    loss_history = []


    model = model.to(DEVICE) # Move model to the designated device (GPU/CPU)
    model.train()  # Set model in training mode

    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad() # Reset gradients for the optimizer
        inputs = inputs.to(DEVICE) # Move inputs to the designated device
        outputs = model(inputs) # Forward pass
        loss = loss_fn(outputs, targets.to(DEVICE)) # Compute the loss
        loss.backward() # Backward pass
        optimizer.step() # Update the model


        # Track some values to compute statistics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=-1)
        all_predictions.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

        # Save loss every 100 batches
        if (i % 100 == 0) and (i > 0):
            running_loss = total_loss / (i + 1)
            loss_history.append(running_loss)

    acc = accuracy_score(all_targets, all_predictions)
    final_loss = total_loss / len(train_loader)
    
    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}, average train accuracy = {acc * 100:.3f}%")
    return acc, final_loss

def test(model, test_loader, loss_fn, epoch=-1):
    """
    Tests a model for one epoch of test data.

    Note:
        In testing and evaluation, we do not perform gradient descent optimization, so steps 2, 5, and 6 are not needed.
        For performance, we also tell torch not to track gradients by using the `with torch.no_grad()` context.

    :param model: PyTorch model
    :param test_loader: PyTorch Dataloader for test data
    :param loss_fn: PyTorch loss function
    :kwarg epoch: Integer epoch to use when printing loss and accuracy

    :returns: Accuracy score
    """
    total_loss = 0
    all_predictions = []
    all_targets = []

    model = model.to(DEVICE)
    model.eval()  # Set model in evaluation mode
    for _, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            outputs = model(inputs.to(DEVICE))
            loss = loss_fn(outputs, targets.to(DEVICE))

            # Track some values to compute statistics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            all_predictions.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    acc = accuracy_score(all_targets, all_predictions)
    final_loss = total_loss / len(test_loader)
    # Print average loss and accuracy
    print(f"Epoch {epoch + 1} done. Average test loss = {final_loss:.2f}, average test accuracy = {acc * 100:.3f}%")
    return acc, final_loss



train_loader, test_loader = get_mnist()

torch.manual_seed(0) # Set the seed for repeatable tests

# Intialize the model, optimizer, and loss function
model = ConvNet() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()

# Train and test the model over the specified number of epochs
train_losses = []
test_losses = []
train_metrics = []
test_metrics = []

for epoch in range(NUM_EPOCHS):
    train_acc, train_loss = train(model, train_loader, loss_fn, optimizer, epoch)
    test_acc, test_loss = test(model, test_loader, loss_fn, epoch)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_metrics.append(train_acc)
    test_metrics.append(test_acc)
    
# Plot the training curve
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(train_losses, c="r", label="Train loss")
axs[0].plot(test_losses, c="b", label="Test loss")
axs[0].legend()
axs[1].set_xlabel("Epochs")

axs[1].plot(train_metrics, "o-", c="r", label="Train accuracy")
axs[1].plot(test_metrics, "o-", c="b", label="Test accuracy")
axs[1].legend()
axs[1].set_xlabel("Epochs")

plt.show()