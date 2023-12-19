from typing import Dict

import torch
import numpy as np
from numpy import asarray
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm.notebook import tqdm

from lstm_model import LSTMClassifier

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.n_epochs = config["n_epochs"]
        self.setup_opt_fn = lambda model: Adam(model.parameters(), 
                                               config["lr"], 
                                               weight_decay=config["weight_decay"])
        self.model = None
        self.opt = None
        self.history = None
        self.threshold = config['model_THR']
        self.loss_fn = BCEWithLogitsLoss()
        self.device = config["device"]
        self.verbose = config.get("verbose", True)

    def fit(self, model, train_loader, val_loader):
        self.model = model.to(self.device)
        self.opt = self.setup_opt_fn(self.model)
        self.history = {"train_loss": [], 
                        "val_loss": [],
                        "val_acc": []}
        for epoch in range(self.n_epochs):
            print(f"   • Epoch {epoch + 1}/{self.n_epochs}:")
            train_info = self._train_epoch(train_loader)
            val_info = self._val_epoch(val_loader)
            self.history["train_loss"].extend(train_info["train_loss"])
            self.history["val_loss"].append(val_info["loss"])
            self.history["val_acc"].append(val_info["acc"])
        return self.model.eval()

    def _train_epoch(self, train_loader):
        self.model.train()
        losses = []
        if self.verbose:
            train_loader = tqdm(train_loader)
        for batch in train_loader:
            self.model.zero_grad()
            texts, labels = batch
            logits = self.model.forward(texts.to(self.device))
            loss = self.loss_fn(logits, labels.to(self.device).float())
            loss.backward()
            self.opt.step()
            loss_val = loss.item()
            if self.verbose:
                train_loader.set_description(f"TRAINING: Loss={loss_val:.3}")
            losses.append(loss_val)
        return {"train_loss": losses}

    def accuracy(self, pred_labels, true_labels, THR):
        pred_lbls = np.where(np.abs(pred_labels.cpu().numpy()) < THR, 1, 0)
        #ac_score = (pred_lbls == true_labels.cpu().numpy()).sum()
        #total = np.shape(true_labels.cpu().numpy())[0]*np.shape(true_labels.cpu().numpy())[1]
        #ac_score = ac_score/total
        ac_score = np.mean(pred_lbls == true_labels.cpu().numpy())
        return ac_score

    def _val_epoch(self, val_loader):
        self.model.eval()
        all_logits = []
        all_labels = []
        if self.verbose:
            val_loader = tqdm(val_loader)
        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch
                logits = self.model.forward(texts.to(self.device))
                val_loss = self.loss_fn(logits, labels.to(self.device).float())
                val_acc = self.accuracy(logits, labels, self.threshold)
                if self.verbose:
                    val_loader.set_description(f"VALIDATING: Loss={val_loss:.3f}; Acc={val_acc:.3f}")

                all_logits.append(logits)
                all_labels.append(labels.float())
        all_labels = torch.cat(all_labels).to(self.device)
        all_logits = torch.cat(all_logits)
        loss = BCEWithLogitsLoss()(all_logits, all_labels).item()
        acc = self.accuracy(all_logits, all_labels, THR=self.threshold)
        if self.verbose:
            val_loader.set_description(f"Loss={loss:.3f}, Acc={acc:.3f}")
        print(f"Validation loss for epoch = {loss:.3f}")
        print(f"Validation accuracy for epoch = {acc:.3f}\n")
        return {"loss": loss, "acc": acc}

    def predict(self, test_loader):
        if self.model is None:
            raise RuntimeError("Model does not exists!")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                texts, _ = batch
                logits = self.model.forward(texts.to(self.device))
                predictions.extend((logits).tolist())
        return asarray(predictions)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Nothing to save!")
        checkpoint = {"num_classes": self.model.num_classes,
                      "dropout_rate": self.model.dropout_rate, 
                      "hidden_size_lstm": self.model.hidden_size_lstm, 
                      "num_layers_lstm": self.model.num_layers_lstm,
                      "dropout_rate_lstm": self.model.dropout_rate_lstm,
                      "bidirectional_lstm": self.model.bidirectional_lstm,
                      "out_size": self.model.out_size,
                      "trainer_config": self.config,
                      "vocab": self.model.vocab,
                      "emb_matrix": self.model.emb_matrix,
                      "state_dict": self.model.state_dict(),
                      "history" : self.history}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path)
        keys = ["num_classes", "dropout_rate", "hidden_size_lstm", 
                "num_layers_lstm", "dropout_rate_lstm", "bidirectional_lstm", 
                "out_size", "trainer_config",
                "vocab", "emb_matrix", "state_dict"]
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")
        new_model = LSTMClassifier(ckpt["num_classes"],
                                   ckpt["dropout_rate"],
                                   ckpt["hidden_size_lstm"],
                                   ckpt["num_layers_lstm"],
                                   ckpt["dropout_rate_lstm"],
                                   ckpt["bidirectional_lstm"],
                                   ckpt["out_size"],
                                   ckpt["vocab"], 
                                   ckpt["emb_matrix"])
        new_model.load_state_dict(ckpt["state_dict"])
        new_trainer = cls(ckpt["trainer_config"])
        new_trainer.model = new_model
        new_trainer.history = ckpt["history"]
        new_trainer.model.to(new_trainer.device)
        return new_trainer
