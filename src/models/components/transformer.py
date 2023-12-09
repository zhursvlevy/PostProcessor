from typing import Dict

import torch
from transformers import AutoModel, XLMRobertaModel


class Regressor(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, p: float = 0.2):
        super().__init__()
        self.regressor = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p),
                torch.nn.Linear(hidden_dim, 1),
                torch.nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

class RegressionTransformer(torch.nn.Module):

    def __init__(self, 
                 model_path: str,
                 input_dim: int,
                 hidden_dim: int,
                 dropout_rate: int,
                 freeze: bool = False):
        super().__init__()
        self.model_name = model_path
        self.encoder = AutoModel.from_pretrained(self.model_name)
        if freeze: self._freeze()
        self.regressor = Regressor(input_dim, 
                                   hidden_dim, 
                                   dropout_rate)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = torch.mean(self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state, dim=1)

        return self.regressor(output)
    
    def _freeze(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False


class RegressionE5(torch.nn.Module):

    def __init__(self, 
                 model_path: str,
                 input_dim: int,
                 hidden_dim: int,
                 dropout_rate: int,
                 freeze: bool = False):
        super().__init__()
        self.model_name = model_path
        self.encoder = XLMRobertaModel.from_pretrained(model_path, use_cache=False)
        if freeze: self._freeze()
        self.regressor = Regressor(input_dim, 
                                   hidden_dim, 
                                   dropout_rate)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = torch.mean(self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state, dim=1)

        return self.regressor(output)
    
    def _freeze(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False


    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]