import sys
import time
import json
import torch
import argparse
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

def val(data_loader, device):
    model.eval()
    val_acc = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = Variable(images.to(device)), Variable(labels.to(device))

            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)

            val_acc += torch.sum(prediction == labels.data).item()

        val_acc = val_acc / 64000

    return val_acc

def train(model, data_loader, device):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]

    for i, (images, labels) in enumerate(data_loader):
        images, labels = Variable(images.to(device)), Variable(labels.to(device))

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        _, prediction = torch.max(outputs.data, 1)
        train_loss += loss.cpu().data * images.size(0)
        train_acc += torch.sum(prediction == labels.data).item()

        sys.stdout.write('\r[Epoch: %d/%d - Batch: %d/%d] LR: %f' % (epoch+1, args.epochs , i, len(data_loader), lr))

    train_loss = train_loss / 64000
    train_acc = train_acc / 64000
    return train_loss, train_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='location of data')
    parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--model_path', type=str, default='weights.pth',
                        help='location of model weights')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1008, metavar='S',
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Load pre-trained model
    model = Model().to(args.device)

    # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer = SGD(model.parameters(), lr=0.01)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True

    print('n parameters: %d' % sum([m.numel() for m in model.parameters()]))

    # Load data
    data_loader_train = load_data(args.data_dir, args.batch_size, split='train')
    data_loader_val = load_data(args.data_dir, 64, split='val')

    best_acc = 0.0
    history_train_acc, history_val_acc, history_train_loss = [], [], []

    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_acc = train(model, data_loader_train, args.device)
            val_acc = val(data_loader_val, args.device)
            print("\nTraining Epoch: %d, Train Acc: %.3f%% , Train Loss: %.4f , Val Acc: %.3f%%, Elapsed: %.1fs" % (epoch+1, train_acc, train_loss, val_acc, time.time() - t0))

            history_train_loss.append(train_loss)
            history_train_acc.append(train_acc)
            history_val_acc.append(val_acc)

            if val_acc > best_acc:
                torch.save(model.state_dict(), 'weights.pth')
                print("Weight Saved")
                best_acc = val_acc
            # elif (epoch > 1) and (val_acc < best_acc):
            #     adjust_learning_rate()
            exp_lr_scheduler.step(val_acc)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    plt.plot(history_val_acc, label='Val Acc')
    plt.plot(history_train_acc, label= 'Train Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
