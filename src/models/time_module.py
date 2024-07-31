from typing import Any, List

import torch
import lightning.pytorch as pl
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import Precision, Recall

import models.augmentation.augment_signal as T

import pandas as pd
import numpy as np


class TSModule(pl.LightningModule):
    """Example of LightningModule for binary classification."""

    def __init__(
        self,
        feature_name: str,
        target_name:   str,
        id_name: str,
        save_dir: str,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs: Any,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.feature_name = feature_name
        self.target_name = target_name

        self.augmenter = T.RandAugment(magnitude=10, augmentation_operations='Random_block')

        self.id_name = id_name
        self.save_dir = save_dir
        self.net = net

        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.train_precision_macro = Precision(task="binary", num_classes=2, average='macro')
        self.val_precision_macro = Precision(task="binary", num_classes=2, average='macro')
        self.test_precision_macro = Precision(task="binary", num_classes=2, average='macro')

        self.train_recall_macro = Recall(task="binary", num_classes=2, average='macro')
        self.val_recall_macro = Recall(task="binary", num_classes=2, average='macro')
        self.test_recall_macro = Recall(task="binary", num_classes=2, average='macro')

        self.train_precision_weighted = Precision(task="binary", num_classes=2, average='weighted')
        self.val_precision_weighted = Precision(task="binary", num_classes=2, average='weighted')
        self.test_precision_weighted = Precision(task="binary", num_classes=2, average='weighted')

        self.train_recall_weighted = Recall(task="binary", num_classes=2, average='weighted')
        self.val_recall_weighted = Recall(task="binary", num_classes=2, average='weighted')
        self.test_recall_weighted = Recall(task="binary", num_classes=2, average='weighted')

        self.val_acc_best = MaxMetric()

        self.all_nounid = []
        self.all_preds = []
        self.all_targets = []
        self.score_x0_0 = []
        self.score_x0_1 = []

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x = batch[self.feature_name]
        y = batch[self.target_name]
        logits = self.forward(x)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        celoss = self.criterion(logits, y)
        loss = celoss
        preds = torch.argmax(logits, dim=1)
        return loss, preds, torch.argmax(y, dim=1), torch.nn.functional.softmax(logits, dim=-1)

    def training_step(self, batch: Any, batch_idx: int):
        batch[self.feature_name] = self.augmenter(batch[self.feature_name], nb_epoch=10)
        loss, preds, targets, scores = self.model_step(batch)
        
        self.train_loss(loss)
        self.train_acc(preds, targets)
        
        self.train_precision_macro(preds, targets)
        self.train_recall_macro(preds, targets)
        self.train_precision_weighted(preds, targets)
        self.train_recall_weighted(preds, targets)
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision_macro", self.train_precision_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall_macro", self.train_recall_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision_weighted", self.train_precision_weighted, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall_weighted", self.train_recall_weighted, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, scores = self.model_step(batch)
        
        self.val_loss(loss)
        self.val_acc(preds, targets)
        
        self.val_precision_macro(preds, targets)
        self.val_recall_macro(preds, targets)
        self.val_precision_weighted(preds, targets)
        self.val_recall_weighted(preds, targets)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision_macro", self.val_precision_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall_macro", self.val_recall_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision_weighted", self.val_precision_weighted, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall_weighted", self.val_recall_weighted, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, scores = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_precision_macro(preds, targets)
        self.test_recall_macro(preds, targets)
        self.test_precision_weighted(preds, targets)
        self.test_recall_weighted(preds, targets)
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision_macro", self.test_precision_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall_macro", self.test_recall_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision_weighted", self.test_precision_weighted, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall_weighted", self.test_recall_weighted, on_step=False, on_epoch=True, prog_bar=True)

        self.all_nounid.extend(np.array(batch['noun_id']).astype('str'))
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        self.score_x0_0.extend(scores[:, 0].cpu().numpy().reshape(-1,))
        self.score_x0_1.extend(scores[:, 1].cpu().numpy().reshape(-1,))

        pd_results = pd.DataFrame({
            'noun_id': self.all_nounid, 
            'label': self.all_targets, 
            'pred': self.all_preds, 
            'score_x0_0': self.score_x0_0, 
            'score_x0_1': self.score_x0_1,
        })
        pd_results.to_csv(f"{self.save_dir}/results_test.csv", index=None)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    torch.manual_seed(1234)
    _ = TSModule(None, None, None, None, None, None)