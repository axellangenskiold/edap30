# Author: Marcus Klang <Marcus.Klang@cs.lth.se>
# A script for a minimal test case of CNN and LSTM functionallity in pytorch
import torch
from torch import nn, optim
import argparse

parser = argparse.ArgumentParser(
                    prog='test-pytorch',
                    description='Tests a PyTorch installation',
                    epilog='')

parser.add_argument('--device',type=str, default="cpu", help='device to use: cpu, cuda (nvidia), mps (macOS)')

class StupidNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Input: (?, 1, 6, 6)
        # Output: (?, 32, 6, 6)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        # Input: (?, 32*6, 6)
        # Output: (?, 32*6, 16)
        self.lstm = nn.LSTM(input_size=6, hidden_size=16)
        self.relu = nn.ReLU()

        # Input(?, 32*6*16)
        # Output(?, 4)
        self.fc1 = nn.Linear(32*6*16, 4)
    
    def forward(self, X):
        out = self.conv2(X)
        out = self.flatten(out)

        # Reshape it to be a proper input to LSTM
        out = out.view(-1,6*32,6)

        # Use the sequence output
        out, _ = self.lstm(out)

        out = self.flatten(out)
        out = self.fc1(out)
        return out

def main(args):
    # batch of 1 with 1 channel and 6x6 pixels
    X_train = torch.arange(1,37).float().view(1,1,6,6)
    y_train = torch.LongTensor([2])

    net = StupidNetwork()

    # Setup optimizer and cross entropy loss
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    # Prepare device
    device = torch.device(args["device"])

    # Move the net to device
    net.to(device)
    net.train()

    # Attempt inference
    y_logits = net(X_train.to(device))

    # Attempt optimizer step
    optimizer.zero_grad()

    # Compute loss
    loss = criterion(y_logits, y_train.to(device))
    
    # Compute gradient through backpropagation
    loss.backward()

    # Apply gradient
    optimizer.step()

    X = torch.randn(384,1024,1024).to(device)
    X.add_(1)
    print(X.sum())

    # Empirically tested to be around 1.4 most of the time, it should give a value is the most important bit!
    if loss.item() >= 1.2 and loss.item() < 2.0:
        print("Pass: ", loss.item())
    else:
        print("Fail: ", loss.item())

if __name__ == "__main__":
    main(vars(parser.parse_args()))

