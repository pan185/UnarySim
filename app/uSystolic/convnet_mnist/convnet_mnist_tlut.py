from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from six.moves import urllib
from UnarySim.kernel.conv import TlutConv2d
from UnarySim.kernel.linear import TlutLinear

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

class Net(nn.Module):
    def __init__(self, state_dict=None, bitwidth=None, sa=True, et_cycle=None):
        super(Net, self).__init__()
        if sa is True:
            param_list = [param for param in state_dict]
            print("load model parameters: ", param_list)
            state_list = [state_dict[param] for param in param_list]
            self.conv1 = TlutConv2d(1, 32, 3, 1, binary_weight=state_list[0], binary_bias=state_list[1], cycle=et_cycle[0], bitwidth=bitwidth[0])
            self.conv2 = TlutConv2d(32, 64, 3, 1, binary_weight=state_list[2], binary_bias=state_list[3], cycle=et_cycle[1], bitwidth=bitwidth[1])
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = TlutLinear(9216, 128, binary_weight=state_list[4], binary_bias=state_list[5], cycle=et_cycle[2], bitwidth=bitwidth[2])
            self.fc2 = TlutLinear(128, 10, binary_weight=state_list[6], binary_bias=state_list[7], cycle=et_cycle[3], bitwidth=bitwidth[3])
        else:
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--inp-bw', type=int, default=5, metavar='inp_bw',
                        help='input bitwidth')
    parser.add_argument('--wght-bw', type=int, default=9, metavar='wght_bw',
                        help='wght bitwidth')
    parser.add_argument('--cycle', type=int, default=15, metavar='cycle',
                        help='early termination cycle')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('/home/zhewen/data/mnist', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/home/zhewen/data/mnist', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    checkpoint = torch.load("mnist_cnn.pt", map_location=device)

    bw_tuple = (args.inp_bw, args.wght_bw)

    bitwidth_list = [bw_tuple for x in range(4)]
    cycle_list = [args.cycle for x in range(4)]


    print("test tlut model without retraining")
    model = Net(state_dict=checkpoint, bitwidth=bitwidth_list, sa=True, et_cycle=cycle_list).to(device)
    test(model, device, test_loader)
    
    print("test pretrained fp model")
    model = Net(state_dict=checkpoint, bitwidth=bitwidth_list, sa=False).to(device)
    model.load_state_dict(checkpoint)
    test(model, device, test_loader)
    

if __name__ == '__main__':
    main()