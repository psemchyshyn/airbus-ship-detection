import torch
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torch.utils import data
import albumentations as A
import tqdm
import pandas as pd
import cv2
import os
import numpy as np
from utility import *
from torchvision.transforms import ToPILImage, ToTensor
from dataset import ShiptDetectionDataset, TestDataset
from skimage.morphology import binary_opening, disk, label

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ShipSegmentModel(pl.LightningModule):
    def __init__(self, config_model, config_training, conf_data):
        super(ShipSegmentModel, self).__init__()
        self.conf_model = config_model
        self.conf_training = config_training
        self.conf_data = conf_data

        self.save_hyperparameters()

        augs = A.OneOf([
                A.VerticalFlip(p=0.5),              
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),], p=0.8)
        

        self.train_ds = ShiptDetectionDataset(self.conf_data["train_images"], 
                                                self.conf_data["train_labels"],
                                                self.conf_data["image_h"], 
                                                self.conf_data["image_w"], 
                                                reduce=self.conf_data["reduce_dataset"],
                                                balance=self.conf_data["balance_dataset"],
                                                split=self.conf_data["train_val_split"],
                                                mode="train",
                                                transforms=augs)
        
        self.val_ds = ShiptDetectionDataset(self.conf_data["train_images"], 
                                                self.conf_data["train_labels"],
                                                self.conf_data["image_h"], 
                                                self.conf_data["image_w"], 
                                                reduce=self.conf_data["reduce_dataset"],
                                                balance=self.conf_data["balance_dataset"],
                                                split=self.conf_data["train_val_split"],
                                                mode="val")

        self.test_ds = TestDataset(self.conf_data["test_images"],                                         self.conf_data["image_h"], 
                                        self.conf_data["image_w"], 
                                        ext=self.conf_data["ext"])
        
        self.model = smp.create_model(self.conf_model["model_name"], self.conf_model["encoder_name"], encoder_depth=self.conf_model["encoder_depth"], encoder_weights=self.conf_model["encoder_weights"], in_channels=self.conf_model["channels"], classes=1)

        self.dice_loss = smp.losses.DiceLoss(smp.losses.constants.BINARY_MODE)
        self.focal_loss = smp.losses.FocalLoss(smp.losses.constants.BINARY_MODE)
        self.lovasz_loss = smp.losses.LovaszLoss(smp.losses.constants.BINARY_MODE)
        self.ce_loss = smp.losses.SoftBCEWithLogitsLoss()

        params = smp.encoders.get_preprocessing_params(self.conf_model["encoder_name"])
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.conf_model["encoder_weights"] is not None:
            x = (x - self.mean) / self.std
        mask = self.model(x)
        return mask
    
    def calc_loss(self, pred, gt):
        loss = self.ce_loss(pred, gt.float())

        if self.conf_training["switch_to_lovasz_loss"] <= self.current_epoch:
            loss += self.lovasz_loss(pred, gt)
        else:
            loss += self.focal_loss(pred, gt)
        return loss

    def shared_step(self, batch, mode):
        x = batch["image"]
        mask = batch["mask"]
        mask = mask.long()
        y = self(x)


        loss = self.calc_loss(y, mask)
        pred_mask = (y > 0.5).float()
        self.log(f"{mode}_loss", loss)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, mask, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log(f"{mode}_iou", iou_score)
        
        return {"loss": loss, "iou_score": iou_score, "image": x, "pred_mask": pred_mask, "gt": mask.float()}

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, "train")
        return {"loss": results["loss"], "iou_score": results["iou_score"]}

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, "val")
        imgs = map(TF.to_pil_image, [results["image"][0], results["pred_mask"][0], results["gt"][0]])
        self.logger.log_image(key=f"epoch-{self.current_epoch}", images=list(imgs), caption=["image", "prediction", "mask"])
        return {"loss": results["loss"], "iou_score": results["iou_score"]}

    def on_test_epoch_start(self):
        self.submission = {'ImageId': [], 'EncodedPixels': []}

    def test_step(self, batch, batch_idx):
        image_ids = batch["image_id"]
        batch_mask = self(batch["image"])
        batch_mask = batch_mask.squeeze(1).cpu().detach().numpy()
        for image_id, mask in zip(image_ids, batch_mask):
            mask = binary_opening(mask > 0.5, disk(2))
            mask = cv2.resize(mask.astype(np.uint8), (self.conf_data["orig_size"],)*2)
            mask = mask.astype(np.bool_)

            labels = label(mask)
            encodings = [mask2rle(labels == k) for k in np.unique(labels[labels > 0])]
            if len(encodings) > 0:
                for encoding in encodings:
                    self.submission['ImageId'].append(image_id)
                    self.submission['EncodedPixels'].append(encoding)
            else:
                self.submission['ImageId'].append(image_id)
                self.submission['EncodedPixels'].append(None)


    def on_test_epoch_end(self):
        submission_df = pd.DataFrame(self.submission, columns=['ImageId', 'EncodedPixels'])
        submission_df.to_csv(self.conf_training["submission_name"], index=False)

    def predict(self, folder, out):
        os.makedirs(out, exist_ok=True)
        for image_name in tqdm.tqdm(os.listdir(folder)):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(f"{image_path}")
            image = cv2.resize(image, (self.conf_data["image_w"], self.conf_data["image_h"]))

            x = ToTensor()(image)
            x = x.unsqueeze(0).to(self.device)
            pred_mask = (self(x) > 0.5).float()
            pred_mask = cv2.resize(pred_mask.squeeze().squeeze(0).detach().cpu().numpy(), (self.conf_data["orig_size"], )*2)
            cv2.imwrite(os.path.join(out, image_name), pred_mask)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf_training["lr"])
        return [optimizer]

    def train_dataloader(self):
        return data.DataLoader(self.train_ds, batch_size=self.conf_training["batch_size"], shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return data.DataLoader(self.val_ds, batch_size=self.conf_training["batch_size"], shuffle=False, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, batch_size=self.conf_training["batch_size"], shuffle=False, pin_memory=True, num_workers=4)
