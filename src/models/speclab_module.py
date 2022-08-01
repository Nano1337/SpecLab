from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, JaccardIndex
import torch.nn.functional as F

# def dice_score(input, target):
#     """Dice Score Metric.

#     :param input: The input (predicted)
#     :param target:  The target (ground truth)
#     :returns: the Dice score between 0 and 1.
#     """
#     smooth = 1.

#     iflat = input.view(-1)
#     tflat = target.view(-1)
#     intersection = (iflat * tflat).sum()
    
#     return ((2. * intersection + smooth) /
#               (iflat.sum() + tflat.sum() + smooth))
class SpecLabLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = F.binary_cross_entropy_with_logits

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_dice = JaccardIndex()
        self.val_dice = JaccardIndex()
        self.test_dice = JaccardIndex()

        # for logging best so far validation dice score
        self.val_dice_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_dice_best doesn't store dice score from these checks
        self.val_dice_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())
        preds = torch.argmax(logits, dim=1).float()
        preds = preds[:, None, :, :]
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        dice = self.train_dice(preds, targets)
        print(f"Dice on batch: {dice}")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/dice", dice, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_dice.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        dice = self.val_dice(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/dice", dice, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        dice = self.val_dice.compute()  # get val dice score from current epoch
        self.val_dice_best.update(dice)
        self.log("val/dice_best", self.val_dice_best.compute(), on_epoch=True, prog_bar=True)
        self.val_dice.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        dice = self.test_dice(preds, targets)
        self.log("test/loss", loss, on_step=True, on_epoch=True)
        self.log("test/dice", dice, on_step=True, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_dice.reset()

    def configure_optimizers(self):
        """Return optimizers and schedulers."""
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, verbose=True)
        return [optimizer], [scheduler]

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "speclab.yaml")
    _ = hydra.utils.instantiate(cfg)
    print("speclab_module: Pass")
