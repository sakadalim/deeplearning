import os
import torch
import torch.nn as nn

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3,out_channels=out_channels, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.relu(output)

        return output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.unit1 = Unit(in_channels=3, out_channels=96)
        self.unit2 = Unit(in_channels=96, out_channels=192)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit3 = Unit(in_channels=192, out_channels=384)
        self.unit4 = Unit(in_channels=384, out_channels=384)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit5 = Unit(in_channels=384, out_channels=768)
        self.unit6 = Unit(in_channels=768, out_channels=1536)

        self.avgpool = nn.AvgPool2d(kernel_size=48)

        self.net = nn.Sequential(self.unit1, self.unit2, self.pool1,
                                 self.unit3, self.unit4, self.pool2,
                                 self.unit5, self.unit6, self.avgpool)


        self.fc= nn.Linear(in_features=1536, out_features=1000)

        # Load pre-trained model

        exists = os.path.isfile('weights.pth')
        if exists:
            print("Loading weights")
            self.load_weights('weights.pth')
        else:
            print("No weights found. Starting Fresh")

    def load_weights(self, pretrained_model_path, cuda=torch.cuda.is_available()):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=torch.cuda.is_available())

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,1536)
        output = self.fc(output)
        return output
