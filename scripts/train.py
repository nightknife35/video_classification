from settings.import_settings import import_settings
settings = import_settings()
import torch
import sqlite3
import numpy as np
from datetime import datetime


def train(model, data_generator, device):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(settings['epochs']):

        model.train() # set the model to train
        n_of_videos = data_generator.n_of_videos()
        running_loss = 0.0
        correct = 0
        total = 0


        #for i in range(n_of_videos):
        for i in range(n_of_videos-10):
            x, y = data_generator.get_video()
            x, y = x.to(device), y.to(device)
            y_pred = model(x) # not just [output values of each node]. Rather a full fucking model
            loss = model.criterion(y_pred, y) # u pass in a full model and somewhere somehow, just like it has model.params(), it has model.grads()
            loss.backward() # when u calculate the gradients, thats where the numbers go (model.grads() aka the same size as my parametre but 0.01)
            if i%settings['batch_size']==0:
                model.optimizer.step()
                model.optimizer.zero_grad() # zeros out the model.grads()

            y_pred = y_pred.squeeze() # from (1 101) to (101)
            running_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 0)
            predicted = predicted.item()
            total += 1
            y_actual = y.squeeze()
            y_actual = torch.nonzero(y_actual)[0][0]
            y_actual = y_actual.item()
            if (predicted == y_actual):
                correct += 1
            """
                y_pred = model(inputs) # (32, 10)
                one_output = y_pred[0] # array[10]
                print(one_output)
                _, predicted = torch.max(one_output, 0)
                predicted = predicted.item()
                print(predicted)
            """

            print(f"done {i+1} vid") # about 10 vids
            del y_pred
            del y_actual
            del x
            del y
            del predicted
            torch.cuda.empty_cache()


        # sudo save
        
        torch.save(model.state_dict(), f'models/timestamp{timestamp}epoch{epoch+1}.pth')

        conn = sqlite3.connect('data/database.db')
        c = conn.cursor()
        c.execute("UPDATE videos SET visited = 0")
        conn.commit()


        epoch_loss = running_loss / n_of_videos
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{settings["epochs"]}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

