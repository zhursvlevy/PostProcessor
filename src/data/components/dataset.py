import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from src.utils.utils import wilson_score


class RateDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 tokenizer: AutoTokenizer,
                 max_seq_len: int):
        dataframe = pd.read_parquet(data_dir).iloc[:32]
        self.text = dataframe['text_markdown'].tolist()
        if "wilson_rate" in dataframe.columns:
            self.targets = dataframe['wilson_rate'].tolist()
        else:
            self.targets = wilson_score(dataframe["pluses"].to_numpy(), 
                                        dataframe["minuses"].to_numpy()).tolist()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

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
        return len(self.text)
