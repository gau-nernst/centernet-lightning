import os

from torch.utils.data import Dataset
import cv2

class KITTITrackingDataset(Dataset):
    def __init__(self, data_dir, sequence_names, transforms=None, name_to_label=None):
        super().__init__()
        self.sequences = []
        self.num_frames = 0

        for name in sequence_names:
            sequence = KITTITrackingSequence(data_dir, name, transforms=transforms, name_to_label=name_to_label)
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

class KITTITrackingSequence(Dataset):
    _default_name_to_label = {
        "Car": 0, 
        "Van": 1, 
        "Truck": 2, 
        "Pedestrian": 3, 
        "Person": 4, 
        "Cyclist": 5, 
        "Tram": 6, 
        "Misc": 7
    }

    def __init__(self, data_dir, sequence_name, transforms=None, name_to_label=None, img_ext=".png"):
        super().__init__()
        self.img_dir = os.path.join(data_dir, "image_02", sequence_name)
        self.transforms = transforms
        self.img_ext = img_ext
        label_path = os.path.join(data_dir, "label_02", f"{sequence_name}.txt")

        self.seq_length = 0
        annotations = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.rstrip().split()
                if line[1] != "-1":
                    annotations.append(line)
                    self.seq_length = max(int(line[0])+1, self.seq_length)
        
        name_to_label = self._default_name_to_label if name_to_label is None else name_to_label
        self.num_tracks = 0
        self.sequence = [{"ids": [], "labels": [], "bboxes": []} for _ in range(self.seq_length)]
        for line in annotations:
            frame = int(line[0])
            track_id = int(line[1])
            label = name_to_label[line[2]]

            x1, y1, x2, y2 = [float(value) for value in line[6:10]]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            if w < 1 or h < 1:
                continue
            box = [cx, cy, w, h]

            self.num_tracks = max(self.num_tracks, track_id+1)
            self.sequence[frame]["ids"].append(track_id)
            self.sequence[frame]["labels"].append(label)
            self.sequence[frame]["bboxes"].append(box)

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, f"{index:06d}{self.img_ext}")
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        frame = self.sequence[index]
        ids = frame["ids"]
        labels = frame["labels"]
        bboxes = frame["bboxes"]

        for box in bboxes:
            box[0] /= width
            box[1] /= height
            box[2] /= width
            box[3] /= height

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
