import os
import configparser

from torch.utils.data import Dataset
import cv2

class MOTTrackingSequence(Dataset):
    # https://motchallenge.net/instructions/

    def __init__(self, data_dir):
        info_path = os.path.join(data_dir, "seqinfo.ini")
        parser = configparser.ConfigParser()
        parser.read(info_path)
        sequence_info = parser["Sequence"]
        
        self.img_dir = os.path.join(data_dir, sequence_info["imDir"])
        self.frame_rate = float(sequence_info["frameRate"])
        self.seq_length = int(sequence_info["seqLength"])
        self.img_width = int(sequence_info["imWidth"])
        self.img_height = int(sequence_info["imHeight"])
        self.img_ext = sequence_info["imExt"]

        label_path = os.path.join(data_dir, "gt", "gt.txt")
        annotations = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.rstrip().split(",")
                annotations.append(line)
        
        self.num_tracks = 0
        self.sequence = [{"id": [], "bboxes": []} for _ in range(self.seq_length)]
        for line in annotations:
            frame = int(line[0])
            id = int(line[1])
            x, y, w, h = [float(value) for value in line[2:6]]
            
            cx = x-1 + w/2      # minus 1 because xy is zero-indexed
            cy = y-1 + h/2
            box = [
                cx / self.img_width,
                cy / self.img_height,
                w  / self.img_width,
                h  / self.img_height
            ]
            
            self.num_tracks = max(self.num_tracks, id)
            self.sequence[frame-1]["id"].append(id-1)
            self.sequence[frame-1]["bboxes"].append(box)

    def __getitem__(self, index):
        frame = index + 1
        img = os.path.join(self.img_dir, f"{frame:06d}{self.img_ext}")
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        sequence = self.sequence[index]
        ids = sequence["id"]
        bboxes = sequence["bboxes"]
        item = {
            "image": img,
            "ids": ids,
            "bboxes": bboxes
        }
        return item

    def __len__(self):
        return self.seq_length
