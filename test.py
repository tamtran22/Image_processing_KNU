import torch
from unet import UNet
from load_data import MammoDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
# import albumentations
# import albumentations.pytorch
import torchvision.transforms as transforms
import torch.optim as optim

def get_loader(image_dir, label_dir, train_percent=0.8):
    transform = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])
    dataset = MammoDataset(
        image_dir=image_dir, #'ISIC2018_Task1-2_Training_Input',
        label_dir=label_dir, #'ISIC2018_Task1_Training_GroundTruth',
        transform = transform,
        image_suffix='jpg',
        label_suffix='jpg'
    )
    # print(len(dataset.image_path_list), len(dataset.label_path_list))
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
            print(index)
            # images = batch[0].to(device)
            # labels = batch[1].to(device)
            # pred_labels = unet(images)
            
            # loss = criterion(labels, pred_labels)
            # binary_labels = (torch.sigmoid(pred_labels) > 0.5).float()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()





if __name__ == '__main__':
    print('Test...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_loader(
        image_dir='./image',
        label_dir='./label',
        train_percent=0.1
    )

    train_unet(
        train_loader, 
        epochs=5, 
        device=device
    )
    print('ffff')