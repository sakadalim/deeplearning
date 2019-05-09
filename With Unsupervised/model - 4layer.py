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

class Tinu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tinu, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels=in_channels, kernel_size=3,out_channels=out_channels, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.relu(output)


        return output
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        #encoder
        self.unit1 = Unit(in_channels=3, out_channels=96) # 96x96x96
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True) #48x48x96
        self.unit2 = Unit(in_channels=96, out_channels=48)#48x48x48
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)#24x24x48
        self.unit3 = Unit(in_channels=48,out_channels=24)#24x24x24
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)#12x12x24
        self.unit4 = Unit(in_channels=24,out_channels=24)#12x12x24



        #decoder
        self.tinu1 = Tinu(in_channels=24,out_channels=24)#12x12x24
        self.unPool1 = nn.MaxUnpool2d(kernel_size=2)#24x24x24
        self.tinu2 = Tinu(in_channels=24,out_channels=48)#24x24x48
        self.unPool2 = nn.MaxUnpool2d(kernel_size=2)#48x48x48
        self.tinu3 = Tinu(in_channels=48,out_channels=96)#48x48x96
        self.unPool3 = nn.MaxUnpool2d(kernel_size=2), #96x96x96
        self.tinu4 = Tinu(in_channels=96,out_channels=3)#96x96x3
        self.sigmoid = nn.Sigmoid()

        exists = os.path.isfile('weights.pth')
        if exists:
            print("Loading weights")
            self.load_weights('weights.pth')
        else:
            print("No weights found. Starting Fresh")


    def forward(self, input):

        output = self.unit1(input)
        output, indices1 = self.pool1(output)
        output = self.unit2(output)
        output, indices2 = self.pool2(output)
        output = self.unit3(output)
        output, indices3 = self.pool3(output)
        output = self.unit4(output)

        output = self.tinu1(output)
        output = self.unPool1(output, indices3)
        output = self.tinu2(output)
        output = self.unPool2(output, indices2)
        output = self.tinu3(output)
        output = self.unPool3(output, indices1)
        output = self.tinu4(output)
        output = self.sigmoid(output)


        return output

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
