# conda activate pytorch_env

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from settings.import_settings import import_settings
settings = import_settings()

from scripts.data_generator import DataGenerator
from scripts.model import Model
from scripts.train import train
# from scripts.eval import evaluate
import torch



# data
train_generator = DataGenerator('train')
test_generator = DataGenerator('test')
val_generator = DataGenerator('val')

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

# train
train(model, train_generator, device)
# evaluate(model, val_generator)



# torch.save(model.state_dict(), 'video_classification_model.pth')

