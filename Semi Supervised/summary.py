from model import AutoEncoder, Model
import torch
from torchsummary import summary


model = AutoEncoder().to(torch.device("cuda"))

print(summary(model, (3,96,96)))
