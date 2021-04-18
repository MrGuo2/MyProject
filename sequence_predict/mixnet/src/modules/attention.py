import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDotAttention(nn.Module):

    def __init__(self, embedding_size, dropout_rate=0.1):
        super(ScaleDotAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, query, key, value, attn_mask=None):
        # query: [batch_size, target_length, key_embedding_size]
        # key: [batch_size, sequence_length, key_embedding_size]
        # value: [batch_size, sequence_length, value_embedding_size]
        # attn_mask: [target_length, sequence_length]
        assert query.size(-1) == key.size(-1)
        # attn_score: [batch_size, target_length, sequence_length]
        attn_score = torch.bmm(query, key.transpose(1, 2))
        if attn_mask is not None:
            assert attn_mask.size() == attn_score.size()[1:]
            # Add input attention mask to attention score.
            attn_score = attn_score + attn_mask.unsqueeze(0)
        # attn_probs: [batch_size, target_length, sequence_length]
        attn_probs = F.softmax(attn_score, dim=-1)
        # outputs: [batch_size, target_length, value_embedding_size]
        outputs = torch.bmm(attn_probs, value)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs)
        return outputs
