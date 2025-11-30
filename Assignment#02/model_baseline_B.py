import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSimilarityModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        # If bidirectional, the output dim is 2 * hidden_dim
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Optional: A projection layer to map LSTM output to a shared space
        self.projection = nn.Linear(self.output_dim, 128)

    def forward_one(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Pack padded sequence could be used here for efficiency, 
        # but for baseline simplicity we just run LSTM on padded data.
        # We take the last hidden state or max pool.
        # Let's use the final hidden state (or concatenation of fwd/bwd final states).
        
        output, (hn, cn) = self.lstm(embeds)
        
        # output: [batch_size, seq_len, num_directions * hidden_dim]
        # hn: [num_layers * num_directions, batch_size, hidden_dim]
        
        # We can take the last time step output, or pool.
        # Max pooling over time is often robust.
        # output.permute(0, 2, 1) -> [batch_size, hidden_dim, seq_len]
        pooled = F.adaptive_max_pool1d(output.permute(0, 2, 1), 1).squeeze(2)
        
        # Project
        projected = self.projection(pooled)
        return projected

    def forward(self, anchor, choice_a, choice_b):
        """
        Args:
            anchor: [batch_size, seq_len]
            choice_a: [batch_size, seq_len]
            choice_b: [batch_size, seq_len]
        Returns:
            logits: [batch_size, 2] where 0-th index is score for A, 1-st for B.
        """
        emb_anchor = self.forward_one(anchor)    # [batch, dim]
        emb_a = self.forward_one(choice_a)       # [batch, dim]
        emb_b = self.forward_one(choice_b)       # [batch, dim]
        
        # Compute Cosine Similarity
        # F.cosine_similarity returns [batch_size]
        sim_a = F.cosine_similarity(emb_anchor, emb_a)
        sim_b = F.cosine_similarity(emb_anchor, emb_b)
        
        # Stack to form logits: [batch_size, 2]
        # We multiply by a scalar (temperature) to sharpen the distribution if needed,
        # but for raw logits we can just stack.
        # However, CrossEntropyLoss expects unnormalized logits.
        # Cosine sim is [-1, 1].
        logits = torch.stack([sim_a, sim_b], dim=1)
        
        # Scale logits to help optimization (temperature scaling)
        logits = logits * 10.0 
        
        return logits
