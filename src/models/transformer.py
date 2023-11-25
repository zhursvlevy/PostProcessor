from typing import Dict

import torch
from transformers import AutoModel


class Regressor(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, p: float = 0.2):
        self.regressor = torch.nn.Sequential(
            [
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Dropout(p),
                torch.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

class RegressionTransformer(torch.nn.Module):

    def __init__(self, model_path: str, config: Dict):
        super().__init__()
        self.model_name = model_path
        self.config = config
        self.dropout_rate = config['dropout_rate']
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.regressor = Regressor(config["input_dim"], 
                                   config["hidden_dim"], 
                                   config["p"])

    def forward(self, input_ids, attention_mask,):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = output[0]
        hidden_state = hidden_state[:, 0]

        return self.regressor(hidden_state)