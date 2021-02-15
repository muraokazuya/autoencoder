import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class AutoEncoder(torch.nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

class MNISTNumed(MNIST):
    def __init__(self, nums=[1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        tmp = []
        for n in nums:
            subdata= [d for d, t in zip(self.data, self.targets) if t == n]
            tmp.extend(subdata)
        self.data = tmp
        #self.data = [d for d, t in zip(self.data, self.targets) for n in nums if t == n]

def train(net, criterion, optimizer, epochs, trainloader, linear=False):
    losses = []
    output_and_label = []
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    net = net.to(device)

    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}, ', end='')
        running_loss = 0.0
        for counter, (img, _) in enumerate(trainloader, 1):
            img=img.to(device)
            optimizer.zero_grad()
            if linear:
                img = img.reshape(-1, 1 * 28 * 28)
            output = net(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print('loss:', avg_loss)
        output_and_label.append((output, img))
    print('finished')
    return output_and_label, losses

def test(net, testloader, linear=False):
    result = []
    diff = []
    org = []

    for item, _ in testloader:
        if linear:
            out = net(item.reshape(-1, 1 * 28 * 28))
            out = out.reshape(-1, 1, 28, 28)
        else:
            out = net(item)
        org.extend(item)
        result.extend(out)
        diff.extend(abs(item - out))
    
    return (org, result, diff)

###Encoder and decoder with fully connected layers and fewer channels
enc1 = torch.nn.Sequential(
    torch.nn.Linear(1 * 28 * 28, 384),
    torch.nn.ReLU(),
    torch.nn.Linear(384, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 8),
    torch.nn.Tanh(),
)

dec1 = torch.nn.Sequential(
    torch.nn.Linear(8, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 384),
    torch.nn.ReLU(),
    torch.nn.Linear(384, 1 * 28 * 28),
    torch.nn.Tanh()
)

###Encoder and decoder with conv layers and fewer channels
enc2 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 3, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(3, 5, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2)
)

dec2 = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(5, 3, kernel_size=2, stride=2),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2),
    torch.nn.Tanh()
)

###Encoder and decoder with conv layers and more channels
enc3 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2)
)

dec3 = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
    torch.nn.Tanh()
)

###Dataloader preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5))])

trainset = MNISTNumed(root='./data', nums=[1], train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

testset = MNISTNumed(root='./data', nums=[1, 8], train=False, transform=transform, download=True)
testloader = DataLoader(testset, batch_size=16, shuffle=True)

###Constract AE (choose enc1/2/3 and dec1/2/3, see above)
net_linear = AutoEncoder(enc1, dec1)

###Training ("1" is trained)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net_linear.parameters(), lr=0.5)
EPOCHS = 100
output_and_label, losses = train(net_linear, criterion, optimizer, EPOCHS, trainloader, linear=True)

###Visualize output
out, org = output_and_label[-1]
imshow(org.reshape(-1, 1, 28, 28))
imshow(out.reshape(-1, 1, 28, 28))

###Testing (input untrained images "1" and "8")
(org, result, diff) = test(net_linear, testloader, linear=True)
imshow(org[0:20])
imshow(result[0:20])
imshow(diff[0:20])