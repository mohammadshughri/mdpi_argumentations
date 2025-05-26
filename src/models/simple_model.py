import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.models.base_model import BaseClassifierModel


class SimpleClassifier(BaseClassifierModel):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        max_length,
        n_layers,
        loss_type,
        optimizer,
        **kwargs
    ):
        super(SimpleClassifier, self).__init__(loss_type, optimizer)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)  # Shape: (batch_size, seq_length, embedding_dim)

        # Average pooling across sequence length dimension
        x = torch.mean(x, dim=1)  # Shape: (batch_size, embedding_dim)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Remove the final relu - you don't want relu on output for classification
        return x
