from torch.utils.data import Dataset

class DetectionForTracking(Dataset):
    def __init__(self, dataset):
        super().__init__()
        # known limitations: the parent dataset must expose bboxes and transforms
        # thus this will not work with Subset
        assert hasattr(dataset, "bboxes")
        assert hasattr(dataset, "transforms")
        self.dataset = dataset
        self.transforms = dataset.transforms

        # accept "ids" as labels and remove transforms in the original dataset
        # this is a hack based on Albumentations implementation. it might break if Albumentations changes its implementation
        if self.transforms is not None:
            self.transforms.processors["bboxes"].params.label_fields.append("ids")      
            dataset.transforms = None
        
        # set a unique track id for each detection box
        self.num_tracks = 0
        self.track_ids = []
        for img_bboxes in dataset.bboxes:
            num_bboxes = len(img_bboxes)
            img_track_ids = [self.num_tracks + i for i in range(num_bboxes)]
            
            self.track_ids.append(img_track_ids)
            self.num_tracks += num_bboxes

    def __getitem__(self, index):
        item = self.dataset[index]
        item["ids"] = self.track_ids[index]

        if self.transforms is not None:
            augmented = self.transforms(**item)
            return augmented
        
        return item

    def __len__(self):
        return len(self.dataset)
