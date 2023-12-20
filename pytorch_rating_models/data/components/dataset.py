from typing import Tuple, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from pytorch_rating_models.utils.utils import wilson_score
import json


class RateDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 index_file: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_seq_len: int,
                 prepend_title: bool = True,
                 target: str = "wilson",
                 scaler: Any = None,
                 use_scaler: bool = False):

        assert split in ("train", "test", "val"), "split must be train, test, val"
        assert target in ("wilson", "raw"), "target must be 'wilson' or 'raw'"

        with open(index_file) as f:
            index = json.load(f)[split]
        dataframe = pd.read_parquet(data_dir)
        dataframe = dataframe[dataframe["id"].isin(index)]
        self.markdown = dataframe['text_markdown'].tolist()
        self.title = dataframe['title'].tolist()
        if target == "wilson":
            self.targets = wilson_score(dataframe["pluses"].to_numpy(), 
                                        dataframe["minuses"].to_numpy())
        else:
            self.targets = dataframe[["pluses", "minuses"]].to_numpy()      
        if use_scaler:
            assert scaler is not None, "if use_scaler is true"
            if  split == "train":
                self.scaler = scaler
                self.scaler.fit(self.targets)
                self.targets = self.scaler.transform(self.targets)
            else:
                self.scaler = scaler
                self.targets = self.scaler.transform(self.targets)
        else:
            self.scaler = None
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prepend_title = prepend_title

    def __getitem__(self, index):
        text = self._build_input(index)
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float32).view(-1)
        }

    def __len__(self) -> int:
        return len(self.markdown)

    def _build_input(self, idx) -> str:
        markdown = str(self.markdown[idx])
        markdown = markdown.split()
        if self.prepend_title:
            title = str(self.title[idx])
            title = title.split()
            text = " ".join(title + markdown)
            return text
        return " ".join(markdown)