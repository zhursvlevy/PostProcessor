from typing import Dict

import torch
from transformers import AutoModel


class Regressor(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, p: float = 0.2):
        super().__init__()
        self.regressor = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Dropout(p),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

class RegressionTransformer(torch.nn.Module):

    def __init__(self, 
                 model_path: str,
                 input_dim: int,
                 hidden_dim: int,
                 dropout_rate: int):
        super().__init__()
        self.model_name = model_path
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.regressor = Regressor(input_dim, 
                                   hidden_dim, 
                                   dropout_rate)

    def forward(self, input_ids, attention_mask,):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = output[0]
        hidden_state = hidden_state[:, 0]

        return self.regressor(hidden_state)