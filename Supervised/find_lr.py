import math
import torch
import sys
import torch.nn as nn
from adamW import AdamW
from model import Model
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler, SGD
from torchvision import datasets, transforms

def load_data(data_dir, batch_size, split):
    """ Method returning a data loader for labeled data """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96, padding=4),
        transforms.RandomRotation(15, resample=False, expand=False, center=None),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    data = datasets.ImageFolder(f'{data_dir}/supervised/{split}', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader


def find_lr( model, data_loader, device, init_value = 1e-8, final_value=10, beta = 0.98):
    num = len(data_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    model.train()

    for i, (images, labels) in enumerate(data_loader):
        batch_num += 1
        images, labels = Variable(images.to(device)), Variable(labels.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        train_loss = loss.cpu().data * images.size(0)

        avg_loss = beta * avg_loss + (1-beta) * train_loss
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()

        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        sys.stdout.write('\r[ %d/%d] LR: %f' % (i, len(data_loader), lr))

    return log_lrs, losses


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
# optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
# optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
optimizer = SGD(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()
data_loader_train = load_data('./data', 24, split='train')
logs,losses = find_lr(model = model, data_loader = data_loader_train, device = device)
plt.plot(logs[10:-5],losses[10:-5])
plt.show()
