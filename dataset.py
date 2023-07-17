import torch
import numpy as np
import pandas as pd
import glob
import albumentations as A
import cv2
import os
from torch.utils.data import Dataset
from utility import rle2mask


class ShiptDetectionDataset(Dataset):
    def __init__(self, path2images, labels_file, image_h, image_w, balance=True, reduce=1, split=.8, mode="train", transforms=None):
        super(ShiptDetectionDataset, self).__init__()
        self.transforms = transforms
        self.image_h = image_h
        self.image_w = image_w
        self.mode = mode
        self.split = split
        self.path2images = path2images
        self.labels_data = pd.read_csv(labels_file)

        self.transforms = transforms
        if balance:
            self.balance()
        self.setup(reduce=reduce)


    def setup(self, reduce):
        ids = self.labels_data["ImageId"].unique()
        ids = ids[:int(len(ids)*reduce)]
        self.labels_data = self.labels_data[self.labels_data["ImageId"].isin(ids)]



        train_ids = ids[:int(len(ids)*self.split)]
        if self.mode == "train":
            self.labels_data = self.labels_data[self.labels_data["ImageId"].isin(train_ids)]
        elif self.mode == "val":
            self.labels_data = self.labels_data[~self.labels_data["ImageId"].isin(train_ids)]
        else:
            raise Exception("The mode should be either train or val!")
        
        self.ids = self.labels_data["ImageId"].unique()

    def balance(self):
        with_ships = self.labels_data[self.labels_data["EncodedPixels"].notnull()]
        unique_images_with_ships = with_ships["ImageId"].unique()
        without_ships = self.labels_data[~self.labels_data["ImageId"].isin(with_ships["ImageId"].unique())]
        ids_without = without_ships["ImageId"].unique()
        remove_ids = ids_without[:(len(without_ships) - len(unique_images_with_ships))]
        self.labels_data = self.labels_data[~self.labels_data["ImageId"].isin(remove_ids)]
        
    def __len__(self):
        return len(self.ids)
    
    def create_mask(self, image_id, shape=(768, 768)):
        all_ships = self.labels_data.loc[self.labels_data["ImageId"] == image_id, "EncodedPixels"].tolist()
        image_mask = np.zeros(shape, dtype=np.uint8)
        if any([isinstance(el, float) for el in all_ships]):
            return image_mask
        else:
            for mask in all_ships:
                image_mask += rle2mask(mask, shape=shape)
            return image_mask

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        image = cv2.imread(f"{self.path2images}/{image_id}")
        mask = self.create_mask(image_id, shape=image.shape[:2])
        image = cv2.resize(image, (self.image_w, self.image_h))
        mask = cv2.resize(mask, (self.image_w, self.image_h))


        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {"image_id": image_id, "image": torch.from_numpy(image).permute(2, 0, 1) / 255, "mask": torch.from_numpy(mask).unsqueeze(0)}

class TestDataset(Dataset):
    def __init__(self, path2images, image_h, image_w, ext="jpg"):
        super(TestDataset, self).__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.path2images = glob.glob(f"{path2images}/*.{ext}")

    def __len__(self):
        return len(self.path2images)

    def __getitem__(self, idx):
        image = cv2.resize(cv2.imread(self.path2images[idx]), (self.image_w, self.image_h))

        return {"image_id": self.path2images[idx].split(os.sep)[-1], "image": torch.from_numpy(image).permute(2, 0, 1) / 255}
