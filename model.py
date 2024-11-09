import mlflow

assert mlflow.__version__ >= "2.0.0"
mlflow.set_tracking_uri("http://127.0.0.1:8081/")

import torch

assert torch.version.__version__ >= "2.0.0"

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import datasets
from torchvision import transforms

assert torch.version.__version__ >= "0.1.0"

from torchinfo import summary


class MNISTMLP(nn.Module):
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        o = F.softmax(self.layers(x), dim=1)
        return o

# i dont know the reason for existance of this function
def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()                               # count number of correct ones

def train(epoch, data_loader, model, loss_fn, optimizer, _logger):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    # i still don't know the reason for having these two
    total_loss = 0.0
    total_correct = 0

    for (data, target) in data_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)

        # calculate the loss
        loss = loss_fn(output, target)
        _logger.log_metric('loss', loss.item(), step=epoch)
        total_loss += loss.item()
        _logger.log_metric('total_loss', total_loss, step=epoch)

        # total correct
        total_correct += correct(output, target)
        _logger.log_metric('total_correct', total_correct, step=epoch)

        # backward propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items

    _logger.log_metric('train_loss', train_loss)
    _logger.log_metric('accuracy', accuracy)

    return train_loss, accuracy

def test(test_loader, model, criterion, _logger):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()

            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    _logger.log_metric('testset-accuracy', 100 * accuracy)

batch_size = 200 # 200 to train faster

data_dir = './data'
print(data_dir)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    # a very short dataset EDA
    for (data, target) in train_loader:
        print('data:', data.size(), 'type:', data.type())
        print('target:', target.size(), 'type:', target.type())

        # pltsize=1
        # plt.figure(figsize=(10*pltsize, pltsize))
        # for i in range(10):
        #     plt.subplot(1,10,i+1)
        #     plt.axis('off')
        #     plt.imshow(data[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
        #     plt.title('Class: '+str(target[i].item()))

        break

    model = MNISTMLP().to(device)
    model_info = summary(model)

    lr = 10e-3

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    epochs = 10

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": lr,
            "loss_function": loss_fn.__class__.__name__,
            # not correct
            "optimizer": optimizer.__class__.__name__,
            "total parameters": model_info.total_params,
            "trainable parameters": model_info.trainable_params
        }
        mlflow.log_params(params)

        for epoch in range(epochs):
            print(f"Training epoch: {epoch+1}")
            loss, acc = train(epoch, train_loader, model, loss_fn, optimizer, _logger=mlflow)

        test(test_loader, model, loss_fn, _logger=mlflow)

        mlflow.pytorch.log_model(model, registered_model_name="mnist_mlp", artifact_path="mnist_mlp")
