import os
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.d = 50
        self.encoder = nn.Sequential(
            nn.Linear(27648, self.d ** 2),
            nn.BatchNorm1d(self.d ** 2),
            nn.ReLU(),
            nn.Linear(self.d**2, self.d * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.d * 2 , self.d ** 2),
            nn.BatchNorm1d(self.d ** 2),
            nn.ReLU(),
            nn.Linear(self.d ** 2, 27648),
            nn.Sigmoid(),
        )

        exists = os.path.isfile('weights_unsup.pth')
        if exists:
            print("Loading weights_unsup")
            self.load_weights('weights_unsup.pth')
        else:
            print("No weights_unsup found. Starting Fresh")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std



    def forward(self, input):
        # mu_logvar = self.encoder(input.view(-1, 27648)).view(-1, 2, self.d)
        # mu = mu_logvar[:, 0, :]
        # logvar = mu_logvar[:, 1, :]
        # output = self.reparameterize(mu, logvar)
        # return self.decoder(output), mu, logvar
        output = self.encoder(input.view(-1,27648))
        return self.decoder(output)


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
        self.encoder = nn.Sequential(*list(self.trained_model.children())[:-5])
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(27648, 1000)

        exists = os.path.isfile('weights_sup.pth')
        if exists:
            print("Loading weights_sup")
            self.load_weights('weights_sup.pth')
        else:
            print("No weights_sup found. Starting Fresh")


    def forward(self, input):
        output = self.encoder(input)
        output = output.view(-1,27648)
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
