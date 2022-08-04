from typing import Any, Dict, Optional, Tuple
import os
from pathlib import Path

import torch
torch.manual_seed(42) # for reproducibility
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from skimage import io

class SRDataset(Dataset):
    ''' specular reflection dataset to be used in pytorch processing'''

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):

        # if batch 
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # fetch image and mask from dataset path
        img_path = os.path.join(self.img_dir, str(idx).zfill(5) + '.png')
        mask_path = os.path.join(self.mask_dir, str(idx).zfill(5) + '.png')
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        sample = {'image': img, 'mask': mask}

        # apply transformations if any
        if self.transform: 
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            mask[mask==255] = 1
            mask = mask[None, :, :]
        
        return img, mask

class SpecLabDataModule(LightningDataModule):
    """LightningDataModule for the Specular Reflection dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        img_dir: str = r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img",
        mask_dir: str = r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_mask",
        train_val_test_split: Tuple[int, int, int] = (7_331, 2_443, 2_444),
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # A.HorizontalFlip(p=0.5), # don't flip image for prediction
        # A.VerticalFlip(p=0.5),
        ToTensorV2()
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 2

    def prepare_data(self):
        prepared_dataset = SRDataset(self.img_dir, 
                                    self.mask_dir, 
                                    transform=self.transforms
                                    )
        return prepared_dataset

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            full_dataset = self.prepare_data()
            
        #     # uncomment this code for everything except predictions
        #     # create test dataset
        #     train_dataset, self.data_test = torch.utils.data.random_split(full_dataset, 
        #                                                                 [self.hparams.train_val_test_split[0]+self.hparams.train_val_test_split[1], 
        #                                                                 self.hparams.train_val_test_split[2]])

        #     # create train and val datasets
        #     self.data_train, self.data_val = torch.utils.data.random_split(train_dataset, [self.hparams.train_val_test_split[0], self.hparams.train_val_test_split[1]])

        # uncomment this code for predictions instead 
        self.data_test = full_dataset

    def train_dataloader(self): 
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "speclab.yaml")
    cfg.img_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img").resolve()
    cfg.mask_dir = Path(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_mask").resolve()
    _ = hydra.utils.instantiate(cfg)
    print("DataModule: Pass")