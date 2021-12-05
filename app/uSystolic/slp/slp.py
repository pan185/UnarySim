from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from six.moves import urllib
from matplotlib import pyplot as plt
import numpy as np
from UnarySim.kernel.utils import tensor_unary_outlier

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)


    def forward(self, x):
        # plt.imsave('fig1.png', x[1,0,:,:], cmap=plt.cm.gray_r)
        # x = torch.round(x*16)/16
        x = torch.flatten(x, 1)
        # print(x.max()); exit()
        output = self.fc1(x)
        return F.softmax(output, dim=1)


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
    print(f'orig max={model.fc1.weight.data.max()}, orig min={model.fc1.weight.data.min()}')
    model.fc1.weight.data = (torch.round(model.fc1.weight.data*128)/128) # for sw inference, weight within [-1, 1] => [-128, 128] 9bit
    tensor_unary_outlier(model.fc1.weight)
    model.fc1.weight.data.mul_(128).clamp_(-127, 127) # for fpga implementation loading, weight within [-127, 127] 8bit
    model.fc1.weight.data.div_(128)
    print(f'orig max={model.fc1.weight.data.max()}, orig min={model.fc1.weight.data.min()}')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.round(data*15)/15
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
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='Retrain new model for testing')
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
        transforms.ToTensor()
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('/home/zhewen/data/mnist', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/home/zhewen/data/mnist', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    if args.retrain == True: 
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
    else: 
        model.load_state_dict(torch.load('mnist_cnn_slp.pt'))
        test(model, device, test_loader)
    
    print(f'max weight={(model.fc1.weight.data.mul(128).clamp(-127, 127)).max()}, \
            min weight={(model.fc1.weight.data.mul(128).clamp(-127, 127)).min()}, \
            mean={(model.fc1.weight.data.mul(128).clamp(-127, 127)).mean()}, \
            median={(model.fc1.weight.data.mul(128).clamp(-127, 127)).median()}')
    np.savetxt('/home/zhewen/Repo/UnarySim/app/uSystolic/slp/weight.txt', (model.fc1.weight.data.mul(128).clamp(-127, 127)), fmt="%d")
    print('Weight saved!')

    if args.save_model and args.retrain:
        torch.save(model.state_dict(), "mnist_cnn_slp.pt")


if __name__ == '__main__':
    main()