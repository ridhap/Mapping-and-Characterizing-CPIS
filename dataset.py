import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import glob
import pickle
import skimage.draw
import cv2
import matplotlib.pyplot as plt

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = glob.glob(os.path.join(image_dir, '*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        pkl_file = img_path.split('png')[0] + 'pkl'
        mask = self.load_mask(pkl_file, image.shape)

        # mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    
    def load_mask(self, mask_path, img_shape):

        pkl_file = open(mask_path, 'rb')
       
        info =  pickle.load(pkl_file)
        mask = np.zeros([img_shape[0], img_shape[1], len(info)], dtype=np.uint8)
        

        for i, p in enumerate(info):
            rr, cc = skimage.draw.disk((int(p['pixel_y']), int(p['pixel_x'])), p['pixel_rad'])
            ht, wt, _ = cv2.imread(p['image_path']).shape

            rr_temp = []
            cc_temp = []
            for r,c in zip(rr, cc):
                if r>=0 and r < ht and c>=0 and c < wt:
                    rr_temp.append(r)
                    cc_temp.append(c)

            mask[rr_temp, cc_temp, i] = 1

        test_mask = np.any(mask, axis=2)
        return np.array(test_mask, dtype=np.float32)