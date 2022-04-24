import os.path as osp
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, auroc

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup


class CodeSearchNetClassifier(pl.LightningModule):
    """
    Classifier Model
    """
    def __init__(self, n_classes: int = 4, steps_per_epoch: int = None, n_epochs: int = None,
                 bert_model_name: str = "bert-base-uncased", initial_wts_dir: str = None):
        super().__init__()
        if initial_wts_dir:
            self.bert = AutoModel.from_pretrained(osp.join(initial_wts_dir, 'bert'))
        else:
            self.bert = AutoModel.from_pretrained(bert_model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)  # hidden size for bert is 768
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """
            Compute loss in the forward step if labels are given
            (Standard in fine-tuning bert)
        """
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        # output = torch.softmax(output, axis=1) -> Not needed as CrossEntropy already applies softmax

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        acc = accuracy(outputs, labels)
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": acc}  # predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        acc = accuracy(outputs, labels)
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        acc = accuracy(outputs, labels)
        
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, prog_bar=True, logger=True) 
        return {"test_loss": loss, "test_accuracy": acc}  # "test_predictions": outputs, "test_labels": labels}

    def training_epoch_end(self, outputs):
        """
          Computing loss at the end of training
        """
        # labels = []
        # predictions = []
        loss = []
        acc = []

        for output in outputs:
            # detach from CPU because we are training using GPU (might cause crash in cluster)
            #labels.append(output["labels"])

            #for output_predictions in output["predictions"]:
            #    predictions.append(torch.argmax(output_predictions))

            loss.append(output["loss"])
            acc.append(output["acc"])
            
        # labels = torch.cat(labels)
        # predictions = torch.stack(predictions)
        acc = torch.as_tensor(acc)

        # acc = accuracy(pred=predictions, target=labels)
        # self.logger.experiment.add_scalar(f"accuracy/Train", acc, self.current_epoch)
        self.logger.experiment.add_scalar(f"accuracy/Train", acc.mean(), self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)  # original recommended fine-tune lr from bert-paper

        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
    def predict(self, batch, batch_idx: int , dataloader_idx: int = None):
        # not the prediction at the moment
        return self(batch)
