import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig


class BertSimilarityModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # Classifier to predict a score from the [CLS] token
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward_one(self, input_ids, attention_mask):
        # input_ids: [batch, seq_len] (should be [CLS] Anchor [SEP] Choice [SEP])
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch, hidden]
        pooled_output = self.dropout(pooled_output)
        score = self.classifier(pooled_output)  # [batch, 1]
        return score

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        """
        Args:
            input_ids_a: Token ids for (Anchor, Choice A) pair.
            attention_mask_a: Mask for (Anchor, Choice A) pair.
            input_ids_b: Token ids for (Anchor, Choice B) pair.
            attention_mask_b: Mask for (Anchor, Choice B) pair.
        Returns:
            logits: [batch_size, 2]
        """
        score_a = self.forward_one(input_ids_a, attention_mask_a)
        score_b = self.forward_one(input_ids_b, attention_mask_b)
        
        # Concatenate scores to form logits [batch, 2]
        logits = torch.cat([score_a, score_b], dim=1)
        return logits
