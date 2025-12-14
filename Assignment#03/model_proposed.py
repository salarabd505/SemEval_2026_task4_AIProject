import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        energy = self.projection(encoder_outputs)  # [batch_size, seq_len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)  # [batch_size, seq_len]
        # Weighted sum of encoder outputs
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, hidden_dim]
        return outputs, weights

class BiLSTMAttentionSimilarityModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Bidirectional output dim is 2 * hidden_dim
        self.lstm_output_dim = hidden_dim * 2
        
        self.attention = SelfAttention(self.lstm_output_dim)
        
        # Projection layer to map to a shared space
        self.projection = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64) # Final embedding size
        )

    def forward_one(self, x):
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        output, (hn, cn) = self.lstm(embeds)
        # output: [batch_size, seq_len, 2 * hidden_dim]
        
        # Apply Attention
        attn_output, attn_weights = self.attention(output) # [batch_size, 2 * hidden_dim]
        
        # Project
        projected = self.projection(attn_output) # [batch_size, 64]
        return projected

    def forward(self, anchor, choice_a, choice_b):
        """
        Args:
            anchor: [batch_size, seq_len]
            choice_a: [batch_size, seq_len]
            choice_b: [batch_size, seq_len]
        Returns:
            logits: [batch_size, 2]
        """
        emb_anchor = self.forward_one(anchor)    # [batch, dim]
        emb_a = self.forward_one(choice_a)       # [batch, dim]
        emb_b = self.forward_one(choice_b)       # [batch, dim]
        
        # Compute Cosine Similarity
        sim_a = F.cosine_similarity(emb_anchor, emb_a)
        sim_b = F.cosine_similarity(emb_anchor, emb_b)
        
        # Stack to form logits: [batch_size, 2]
        logits = torch.stack([sim_a, sim_b], dim=1)
        
        # Scale logits (temperature)
        logits = logits * 10.0 
        
        return logits
