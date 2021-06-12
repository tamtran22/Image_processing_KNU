import torch
from torch._C import device
from unet import UNet
from load_data import MammoDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import albumentations
import albumentations.pytorch
import torch.optim as optim

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
    return train_loader, test_loader

def train_unet(train_loader, epochs=10, device=None):
    unet = UNet(
        in_channels=3,
        out_channels=1
    )
    optimizer = optim.Adam(
        unet.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        unet.train()
        for index, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            pred_labels = unet(images)
            
            loss = criterion(labels, pred_labels)
            binary_labels = (torch.sigmoid(pred_labels) > 0.5).float()





if __name__ == '__main__':
    print('Test...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
