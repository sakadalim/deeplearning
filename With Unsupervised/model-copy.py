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
        self.unit2 = Unit(in_channels=96, out_channels=96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True) #48x48x96
        self.unit3 = Unit(in_channels=96, out_channels=48)#48x48x48
        self.unit4 = Unit(in_channels=48, out_channels=48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)#24x24x48
        self.unit5 = Unit(in_channels=48,out_channels=24)#24x24x24
        self.unit6 = Unit(in_channels=24, out_channels=24)
        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)#12x12x24
        self.unit7 = Unit(in_channels=24,out_channels=16)#12x12x16
        self.unit8 = Unit(in_channels=16, out_channels=16)



        #decoder
        self.tinu1 = Tinu(in_channels=16,out_channels=16)#12x12x16
        self.tinu2 = Tinu(in_channels=16, out_channels=24)#12x12x24
        self.unPool1 = nn.MaxUnpool2d(kernel_size=2)#24x24x24
        self.tinu3 = Tinu(in_channels=24,out_channels=24)#24x24x96
        self.tinu4 = Tinu(in_channels=24, out_channels=48)#24x24x48
        self.unPool2 = nn.MaxUnpool2d(kernel_size=2)#48x48x48
        self.tinu5 = Tinu(in_channels=48,out_channels=48)#48x48x48
        self.tinu6 = Tinu(in_channels=48, out_channels=96)#48x48x96
        self.unPool3 = nn.MaxUnpool2d(kernel_size=2) #96x96x96
        self.tinu7 = Tinu(in_channels=96,out_channels=96)#96x96x96
        self.tinu8 = Tinu(in_channels=96, out_channels=3)#96x96x3
        self.sigmoid = nn.Sigmoid()

        exists = os.path.isfile('weights_unsup.pth')
        if exists:
            print("Loading weights_unsup")
            self.load_weights('weights_unsup.pth')
        else:
            print("No weights_unsup found. Starting Fresh")


    def forward(self, input):

        output = self.unit1(input)
        output = self.unit2(output)
        output, indices1 = self.pool1(output)
        output = self.unit3(output)
        output = self.unit4(output)
        output, indices2 = self.pool2(output)
        output = self.unit5(output)
        output = self.unit6(output)
        output, indices3 = self.pool3(output)
        output = self.unit7(output)
        output = self.unit8(output)


        output = self.tinu1(output)
        output = self.tinu2(output)
        output = self.unPool1(output, indices3)
        output = self.tinu3(output)
        output = self.tinu4(output)
        output = self.unPool2(output, indices2)
        output = self.tinu5(output)
        output = self.tinu6(output)
        output = self.unPool3(output, indices1)
        output = self.tinu7(output)
        output = self.tinu8(output)
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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.trained_model = AutoEncoder()
        for param in self.trained_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(2304, 1000)

        exists = os.path.isfile('weights_sup.pth')
        if exists:
            print("Loading weights_sup")
            self.load_weights('weights_sup.pth')
        else:
            print("No weights_sup found. Starting Fresh")


    def forward(self, input):
        output = list(self.trained_model.children())[0](input)
        output = list(self.trained_model.children())[1](output)
        output, _= list(self.trained_model.children())[2](output)
        output = list(self.trained_model.children())[3](output)
        output = list(self.trained_model.children())[4](output)
        output, _ = list(self.trained_model.children())[5](output)
        output = list(self.trained_model.children())[6](output)
        output = list(self.trained_model.children())[7](output)
        output, _ = list(self.trained_model.children())[8](output)
        output = list(self.trained_model.children())[9](output)
        output = list(self.trained_model.children())[10](output)
        output = output.view(-1,2304)
        output = self.fc(output)
        return output

    def load_weights(self, pretrained_model_path, cuda=torch.cuda.is_available()):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained s in current model
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
