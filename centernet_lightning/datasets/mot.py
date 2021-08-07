import os
import configparser

from torch.utils.data import Dataset
import cv2

class MOTTrackingDataset(Dataset):
    def __init__(self, data_dir, sequence_names, transforms=None):
        super().__init__()
        self.sequences = []
        self.num_frames = 0

        for name in sequence_names:
            sequence = MOTTrackingSequence(data_dir, name, transforms=transforms)
            self.sequences.append(sequence)
            self.num_frames += len(sequence)
            
    def __getitem__(self, index):
        # find which sequence the index belongs to
        # shift track id to the appropriate range
        track_id_offset = 0

        for sequence in self.sequences:
            if index < len(sequence):
                item = sequence[index]
                item["ids"] = [x + track_id_offset for x in item["ids"]]
                
                return item

            index -= len(sequence)
            track_id_offset += sequence.num_tracks

    def __len__(self):
        return self.num_frames

class MOTTrackingSequence(Dataset):
    # https://motchallenge.net/instructions/

    def __init__(self, data_dir, sequence_name, transforms=None):
        super().__init__()
        info_path = os.path.join(data_dir, sequence_name, "seqinfo.ini")
        parser = configparser.ConfigParser()
        parser.read(info_path)
        sequence_info = parser["Sequence"] 
        
        self.img_dir = os.path.join(data_dir, sequence_name, sequence_info["imDir"])
        self.transforms = transforms
        self.frame_rate = float(sequence_info["frameRate"])
        self.seq_length = int(sequence_info["seqLength"])
        self.img_width = int(sequence_info["imWidth"])
        self.img_height = int(sequence_info["imHeight"])
        self.img_ext = sequence_info["imExt"]

        label_path = os.path.join(data_dir, sequence_name, "gt", "gt.txt")
        annotations = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.rstrip().split(",")
                annotations.append(line)
        
        self.num_tracks = 0
        self.sequence = [{"ids": [], "labels": [], "bboxes": []} for _ in range(self.seq_length)]
        for line in annotations:
            # skip non-person objects
            if int(line[6]) == 0:
                continue

            # frame, id, and xy are 1-index
            frame = int(line[0])
            track_id = int(line[1])
            x, y, w, h = [float(value) for value in line[2:6]]

            x1 = x - 1
            y1 = y - 1
            x2 = x1 + w
            y2 = y1 + h
            x1 = min(self.img_width-1, max(0, x1))
            y1 = min(self.img_height-1, max(0, y1))
            x2 = min(self.img_width-1, max(0, x2))
            y2 = min(self.img_height-1, max(0, y2))

            if x2-x1 < 1 or y2-y1 < 1:
                continue

            # convert to normalized cxcywh
            cx = (x1 + x2) / 2 / self.img_width
            cy = (y1 + y2) / 2 / self.img_height
            w = (x2 - x1) / self.img_width
            h = (y2 - y1) / self.img_height
            box = [cx, cy, w, h]
            
            self.num_tracks = max(self.num_tracks, track_id)
            self.sequence[frame-1]["ids"].append(track_id-1)
            self.sequence[frame-1]["labels"].append(0)      # MOT only has 1 class
            self.sequence[frame-1]["bboxes"].append(box)

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, f"{index + 1:06d}{self.img_ext}")
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        frame = self.sequence[index]
        ids = frame["ids"]
        labels = frame["labels"]
        bboxes = frame["bboxes"]
        
        if self.transforms is not None:
            augmented = self.transforms(image=img, bboxes=bboxes, labels=labels, ids=ids)
            return augmented
        
        item = {
            "image": img,
            "ids": ids,
            "labels": labels,
            "bboxes": bboxes
        }
        return item

    def __len__(self):
        return self.seq_length
