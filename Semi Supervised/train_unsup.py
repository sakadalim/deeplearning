import sys
import time
import json
import torch
import argparse
import torch.nn as nn
from model import AutoEncoder
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler, SGD
from torchvision import datasets, transforms
from torch import autograd


def load_data(data_dir, batch_size):
    """ Method returning a data loader for labeled data """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    data = datasets.ImageFolder(f'{data_dir}/unsupervised', transform=transform)
    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    return data_loader

def loss_function(x_hat, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(
        x_hat, x.view(-1, 27648), reduction='sum'
    )

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + 0.1*KLD

def train(model, data_loader, device):
    model.train()
    train_loss = 0.0
    lr = 0.0
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]

    for i, (images, _) in enumerate(data_loader):
        images = Variable(images.to(device))


        outputs = model(images)

        loss = loss_fn(outputs, images.view(-1,27648))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() == float("inf"):
            print("\n")
            print("INFINITY")
        # train_loss += loss.cpu().data * images.size(0)
        train_loss += (loss.item()/len(data_loader))


        sys.stdout.write('\r[Epoch: %d/%d - Batch: %d/%d] LR: %f' % (epoch+1, args.epochs , i, len(data_loader), lr))

    # train_loss = train_loss  / len(data_loader)
    return train_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='location of data')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--model_path', type=str, default='weights_unsup.pth',
                        help='location of model weights_unsup.pth')
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
    model = AutoEncoder().to(args.device)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = SGD(model.parameters(), lr=0.01)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=0.001)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.BCELoss()
    torch.backends.cudnn.benchmark = True

    print('n parameters: %d' % sum([m.numel() for m in model.parameters()]))

    # Load data
    data_loader_train = load_data(args.data_dir, args.batch_size)


    lowest_loss = float("inf")
    history_train_loss = []

    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train(model, data_loader_train, args.device)
            print("\nTraining Epoch: %d, Train Loss: %.4f, Elapsed: %.1fs" % (epoch+1, train_loss, time.time() - t0))

            history_train_loss.append(train_loss)

            if train_loss < lowest_loss:
                torch.save(model.state_dict(), 'weights_unsup.pth')
                print("Weight Saved")
                lowest_loss = train_loss
            exp_lr_scheduler.step(train_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    plt.plot(history_train_loss, label='Train Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
