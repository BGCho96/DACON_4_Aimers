from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os

def create_classification_dataloader(config, mode="train"):
    assert mode in ["train", "val", "test", "all"], f"Unsupported mode: {mode}"
    
    image_size = config['dataset'].get('image_size', 224)
    data_root = config['dataset']['data_dir']  # e.g. datasets/images-classification
    data_dir = os.path.join(data_root, mode)   # e.g. datasets/images-classification/train
    data_dir = data_root if mode == "all" else os.path.join(data_root, mode)


    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=(mode == "train") and config['train'].get('shuffle', True),
        num_workers=2
    )

    return loader, dataset.classes