from settings.import_settings import import_settings
settings = import_settings()
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # 224x224 --> 112x112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # 112x112 --> 56x56
        self.lstm = nn.LSTM(64 * int(settings['img_width']/4) * int(settings['img_height']/4) , 128, batch_first=True) # prev: 64 * 56 * 56,
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 101)
        
        # -------------------------------------------------------
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss() 
        # -------------------------------------------------------


    def forward(self, x):
        x = x.unsqueeze(0) # inputs are (None, 224, 224, 3). i make them into (1, None, 224, 224, 3)

        batch_size, timesteps, height, width, channels = x.size() # 1 240 256 256 3
        print(timesteps)
        x = x.view(batch_size * timesteps, channels, height, width) # 240 256 256 3

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x) # 240 64 64 64
        
        x = x.view(batch_size, timesteps, -1) # 1, 240, 262144(64*64*64)
        x, _ = self.lstm(x) # 1, 240, 128
        x = x[:, -1, :]  # Take the last output of the LSTM (1, 128)

        x = self.dropout(x) # 1 128
        x = self.fc(x) # 1 101
        return x


