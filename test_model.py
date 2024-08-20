import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from settings.import_settings import import_settings
settings = import_settings()

from scripts.model import Model
from scripts.data_generator import DataGenerator
import torch


def load_model(model_path):
    model = Model()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model




model_path = 'models/timestamp20240820_131611epoch3.pth' #/home/nightknife35/projects/note-app/v1-cude-and-pytorch/
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model_path)

data_generator = DataGenerator("val")
test_vids = []
test_labels = []
for i in range(15):
    x, y = data_generator.get_video()
    test_vids.append(x)
    test_labels.append(y)

for i in range(15):
    # for this to work u have to modify the get_video() function from the DataGenerator to return  "return path, lable"
    print("Video Path: ", test_vids[i],"\nLabel: " ,test_labels[i])






