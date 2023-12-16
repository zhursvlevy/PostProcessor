from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pytorch_models.data.components.dataset import RateDataset


class RateDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        index_file: str,
        tokenizer: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_seq_len: int = 512,
        prepend_title: bool = True,
        target: str = None,
        scaler: Any = None,
        use_scaler: bool = False
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)
            self.data_train = RateDataset(self.hparams.data_dir,
                                          self.hparams.index_file,
                                          "train", 
                                          tokenizer,
                                          self.hparams.max_seq_len,
                                          self.hparams.prepend_title,
                                          self.hparams.target,
                                          scaler=self.hparams.scaler,
                                          use_scaler=self.hparams.use_scaler)
            scaler = self.data_train.scaler
            self.data_val = RateDataset(self.hparams.data_dir, 
                                        self.hparams.index_file,
                                        "val",
                                        tokenizer,
                                        self.hparams.max_seq_len,
                                        self.hparams.prepend_title,
                                        self.hparams.target,
                                        scaler=scaler,
                                        use_scaler=self.hparams.use_scaler)
            
            self.data_test = RateDataset(self.hparams.data_dir, 
                                         self.hparams.index_file,
                                         "test",
                                         tokenizer,
                                         self.hparams.max_seq_len,
                                         self.hparams.prepend_title,
                                         self.hparams.target,
                                         scaler=scaler,
                                         use_scaler=self.hparams.use_scaler)
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
