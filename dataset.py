import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob

class PairImagesDataset(Dataset):
    def __init__(self, datasource, crop, transform=None):

        self.datasource = datasource
        self.transform = transform
        self.crop = crop

        self.flare_path = sorted(glob.glob(self.datasource + '/Flare' + '/*.*'))
        self.gt_path = sorted(glob.glob(self.datasource + '/Bg24k' + '/*.*'))
        
        print("flare Image Loaded with examples:", len(self.flare_path))

    def __len__(self):
        return len(self.flare_path)
    
    def __getitem__(self, idx):
        flare_img = Image.open(self.flare_path[idx])
        gt_img = Image.open(self.gt_path[idx])

        # 检查文件名是否相同
        if os.path.basename(self.flare_path[idx]) != os.path.basename(self.gt_path[idx]):
            raise ValueError('The flare and gt images are not matched.')

        # i, j, h, w = transforms.RandomCrop.get_params(flare_img, output_size=(self.crop, self.crop))
        # flare_img = transforms.functional.crop(flare_img, i, j, h, w)
        # gt_img = transforms.functional.crop(gt_img, i, j, h, w)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.crop, self.crop))
        ])
        flare_img = self.transform(flare_img)
        gt_img = self.transform(gt_img)

        return  gt_img, flare_img
    
class ValImgDataset(Dataset):
    def __init__(self, datasource, transform=None):
        self.data_source = datasource
        self.transform = transform

        self.flare_path = sorted(glob.glob(self.data_source + '/Flare' + '/*.*'))
        self.gt_path = sorted(glob.glob(self.data_source + '/Bg24k' + '/*.*'))
        print("Val Image Loaded with examples:", len(self.flare_path))

    def __len__(self):
        return len(self.flare_path)
    
    def __getitem__(self, idx):
        flare_img = Image.open(self.flare_path[idx])
        gt_img = Image.open(self.gt_path[idx])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        flare_img = self.transform(flare_img)
        gt_img = self.transform(gt_img)

        return gt_img, flare_img

class SingleDataset(Dataset):
    def __init__(self, datasource, outsize, transform=None):
        
        self.datasource = datasource
        self.transform = transform
        self.outsize = outsize

        self.img_path = glob.glob(datasource + '/*.*')

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img_path = self.img_path[idx]

        img = Image.open(img_path)

        self.transform = transforms.Compose([transforms.Resize((self.outsize, self.outsize)),
                                             transforms.ToTensor()])

        img = self.transform(img)

        return img
