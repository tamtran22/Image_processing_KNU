from unet import UNet
from load_data import MammoDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import albumentations
import albumentations.pytorch

def get_loader(train_percent=0.8):
    transform = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.VerticalFlip(p=0.5),
        albumentations.pytorch.ToTensorV2()
    ])
    dataset = MammoDataset(
        image_dir='ISIC2018_Task1-2_Training_Input',
        label_dir='ISIC2018_Task1_Training_GroundTruth',
        transform = transform
    )
    n_train = int(train_percent * len(dataset))
    n_test = len(dataset) - n_train
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[n_train, n_test])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False
    )

def train_unet(data_loader, train_percent=0.8):
    net = UNet(

    )


if __name__ == '__main__':
    print('Test...')
