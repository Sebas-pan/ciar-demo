"""ImageNet transforms for VGG16 classification."""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(imgsz=224, augment=False, is_train=False):
    # No augmentation per user request (augment=False)
    if is_train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(imgsz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    # Deterministic for val/test and for train when augment=False
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
