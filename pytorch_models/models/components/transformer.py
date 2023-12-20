from typing import Dict

import torch
from transformers import AutoModel, AutoConfig


class Regressor(torch.nn.Module):

    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_size: int, 
                 p: float = 0.2):
        super().__init__()
        self.regressor = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p),
                torch.nn.Linear(hidden_dim, output_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

class RegressionTransformer(torch.nn.Module):

    def __init__(self, 
                 model_path: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_size: int,
                 dropout_rate: int,
                 freeze: bool = False,
                 weights_path: str = None,
                 use_sigmoid: bool = True):
        super().__init__()
        self.model_name = model_path
        self.regressor = Regressor(input_dim, 
                                   hidden_dim, 
                                   output_size,
                                   dropout_rate)
        
        self.output_size = output_size
        if weights_path:
            config = AutoConfig.from_pretrained(model_path)
            self.encoder = AutoModel.from_config(config)
            self.load_state_dict(torch.load(weights_path))
        else:
            self.encoder = AutoModel.from_pretrained(self.model_name)
        if freeze: self._freeze()
        if use_sigmoid:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = torch.mean(self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state, dim=1)

        return self.activation(self.regressor(output))
    
    def _freeze(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
