from glob import glob
from numpy.core.fromnumeric import shape
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
import numpy as np

class MammoDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_suffix='jpg', \
            label_suffix='png', transform=None):
        self.transform = transform
        self.image_path_list = glob(f'{image_dir}/*.{image_suffix}')
        self.label_path_list = glob(f'{label_dir}/*.{label_suffix}')
    
    def __getitem__(self, index):
        image, label = read_data_by_index(self, index)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return {'image' : image, 'label': label}
    
    def __len__(self):
        return len(self.image_path_list)

def read_data_by_index(self, index):
    image_path = self.image_path_list[index]
    label_path = self.label_path_list[index]
    image = read_image(image_path)
    label = read_image(label_path)
    image = np.transpose(image, axes=(1,2,0))
    label = np.transpose(label, axes=(1,2,0))
    return image, label

if __name__ == '__main__':
    print('Test dataset...')
    dataset = MammoDataset(
        image_dir='ISIC2018_Task1-2_Training_Input',
        label_dir='ISIC2018_Task1_Training_GroundTruth'
    )
    
    image, label = dataset.__getitem__(19)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
    print(np.shape(image), np.shape(label))

    print(next(enumerate(data_loader)).shape)