import sys
import yaml
import torch
import argparse
from torch import nn
from CDSAgent import CDSAgent

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels, embedding_dim=d_model, padding_idx=0), 
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t:torch.Tensor) -> torch.Tensor:
        emb = self.condEmbedding(t)
        return emb

def get_config(file_name):
    with open(file_name, 'r') as f:
        config = yaml.safe_load(f)
    return config

class ProtoEmbedding():
    """
    A proto learning module for diffusing source to target.
    """
    def __init__(self, params:argparse.Namespace):
        self.params = params
        self.config = get_config('./config/proto.yml')
        self.agent = CDSAgent(self.config, self.params)

    def get_embedding(self) -> torch.Tensor:

        return self.agent.run()
    
    def get_loader(self) -> torch.Tensor:
        
        return self.agent.datasets()

