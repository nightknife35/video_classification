from settings.import_settings import import_settings
settings = import_settings()
import sqlite3
import numpy as np
import cv2
import torch

class DataGenerator:
    def __init__(self, split):
        self.split = split
        self.number_of_idxes = 9

        self.db_path = 'data/database.db'
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

    def n_of_videos(self):
        self.c.execute("SELECT COUNT(*) FROM videos WHERE visited = 0 AND split = ?", (self.split, ))
        vid_count = self.c.fetchone()[0]
        print(f"Number of videos: {vid_count}")
        return vid_count

    def get_video(self):

        self.c.execute(f"SELECT path, label FROM videos WHERE split = ? AND visited = 0 ORDER BY RANDOM() LIMIT 1", (self.split, ))
        res = self.c.fetchall()
        if res:
            for o in res:
                path = o[0]
                lable = o[1]
                self.c.execute("UPDATE videos SET visited = 1 WHERE path = ?", (path,))
            self.conn.commit()

            vid = self.extract_frames(path) # python array of stuff (3, 224, 224, 3)
            vid = torch.from_numpy(vid)
            mapped_lable = settings['mapings'].get(lable, None)
            one_hot_label = np.eye(settings['output_classes'], dtype=np.float32)[mapped_lable]
            label = torch.tensor(one_hot_label)
            label = label.unsqueeze(0)
            return vid, label
        else:
            self.conn.close()
            raise ValueError("tryed getting one too many items")

    def extract_frames(self, path):
        # if u use cut/pad. make the database have time as well, for effeciency

        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            frames.append(frame)
        cap.release()
        frames = np.array(frames, dtype=np.float32)
        return frames

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frame = cv2.resize(frame, (settings['img_height'], settings['img_width']))
        return frame





# perhaps a class and u call a method each time to get the next batch
# output = x(video), y where x is (batches, frames, h, w, c) and y is class

"""
DataGenerator_ = DataGenerator('train')
x, y = DataGenerator_.get_item(0)
print(len(x),len(y))


train_dataset = VideoDataset(video_dir='path/to/train/videos', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# act as if train_loader == full dataset [5, 32, 1, ...]
"""
