import albumentations as A
from albumentations.pytorch import ToTensorV2

def train():
    # Augmentation
    # use albumenations to take care of bbox transform
    # random flip, random scaling (0.6 - 1.3), color jittering
    # centernet uses 0.4 for color jittering
    # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py#L222
    # original centernet also uses PCA augmentation from cornernet, though it was not mentioned in their paper. The first PCA augmentation appeared in AlexNet https://dl.acm.org/doi/pdf/10.1145/3065386. Albumentations also implements this as FancyPCA
    train_augment = A.Compose([
        A.HorizontalFlip(),
        A.RandomScale(),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco'))