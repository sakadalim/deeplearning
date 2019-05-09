from model import Model
import torch
from torchsummary import summary


model = Model().to(torch.device("cuda"))

print(summary(model, (3,96,96)))
