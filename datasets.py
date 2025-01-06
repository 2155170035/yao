import os,time,cv2
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
# 定义dataset
class G1G2Dataset(torch_data.Dataset):
    def __init__(self, ReadColorImage, mode, test_data= 'MDvsFA'):
        self.mode = mode
        self.readcolorimage=ReadColorImage
        self.test_data =test_data
        self.img_size=128
        #不存在的图片
        self.missingimage=[239, 245, 260, 264, 2553, 2561, 2808, 2817, 2819, 3503, 3504, 3947, 3949,
                            3962, 7389, 7395, 8094, 8105, 8112, 8757, 8772]
        self.org=  transforms.Compose([
            transforms.CenterCrop(size=self.img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
        self.gt = transforms.Compose([
            transforms.CenterCrop(size=self.img_size),
            transforms.ToTensor(),
            # transforms.Normalize(((0.5,), (0.5,)) if needed
        ])
        
        if self.mode == 'train':
            self.imageset_dir = os.path.join('./Train/image/')
            self.imageset_gt_dir = os.path.join('./Train/mask/')
        elif self.mode == 'test':
            if self.test_data == 'MDvsFA':
                self.imageset_dir = os.path.join('./Test/MDvsFA/image/')
                self.imageset_gt_dir = os.path.join('./Test/MDvsFA/mask/')
            else:
                self.imageset_dir = os.path.join('./Test/SIRST/image/')
                self.imageset_gt_dir = os.path.join('./Test/SIRST/mask/')
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode == 'train':
            return 10000 
        elif self.mode == 'test':
            if self.test_data==  'MDvsFA':
                return 100 
            else:
                return 427                
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            if idx in self.missingimage:
                idx = 0
            img_dir = os.path.join(self.imageset_dir, "%06d.png"%(idx))
            gt_dir = os.path.join(self.imageset_gt_dir, "%06d.png"%(idx))
            input_images = Image.open(img_dir)
            output_images = Image.open(gt_dir)
            sample_info = {}
            sample_info['input_images'] = self.org(input_images)
            sample_info['output_images'] = self.gt(output_images)

            return sample_info
               
        elif self.mode == 'test':
            if self.test_data=="MDvsFA":
                img_dir = os.path.join(self.imageset_dir, "%05d.png"%(idx))
                gt_dir = os.path.join(self.imageset_gt_dir, "%05d.png"%(idx))
            elif self.test_data=="SIRST":
                if idx == 0:
                    idx = 427
                img_dir = os.path.join(self.imageset_dir, "Misc_%d.png"%(idx))
                gt_dir = os.path.join(self.imageset_gt_dir, "Misc_%d_pixels0.png"%(idx))
            else:
                raise NotImplementedError
            input_images = Image.open(img_dir)
            output_images = Image.open(gt_dir)

            sample_info = {}
            sample_info['input_images'] = self.org(input_images)
            sample_info['output_images'] = self.gt(output_images)

            return sample_info
        else:
            raise NotImplementedError
