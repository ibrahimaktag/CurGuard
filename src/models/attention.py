"""Multi-head self-attention module for hybrid CNN-LSTM network traffic classifiers.

Replaces the original single-head dot-product attention with PyTorch's built-in
nn.MultiheadAttention, enabling the model to jointly attend to multiple
representation subspaces in parallel — a critical upgrade for distinguishing
fine-grained differences between attack categories (e.g. DDoS-ICMP vs DDoS-UDP).
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention pooling over an LSTM output sequence.

    Applies nn.MultiheadAttention (query=key=value=x) to the full
    Bi-LSTM output sequence, then mean-pools the attended representations
    into a single fixed-size context vector passed to the classifier head.

    Shape contract:
        Input  : [Batch, Seq_Len, Hidden_Dim]   (batch_first=True, matches LSTM)
        Output : context      [Batch, Hidden_Dim]
                 attn_weights [Batch, Seq_Len, Seq_Len]  (avg over heads)

    The full attention weight matrix (Seq_Len x Seq_Len) is returned instead
    of the old 1-D weight vector, enabling richer Attention Heatmap
    visualisations during Phase-5 evaluation.

    Args:
        hidden_dim: Embedding dimension fed into the attention layer.
                    Must be divisible by num_heads.
        num_heads:  Number of parallel attention heads.
                    - CloudSpecialist : 4 heads  (256 / 4 = 64 dim per head)
                    - IoTSpecialist   : 4 heads  (128 / 4 = 32 dim per head)
                    Both fit comfortably within 8 GB VRAM on a 3050 Ti.
        dropout:    Dropout probability applied to attention weights during
                    training (0.0 = disabled, safe default for small datasets).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads}). "
                f"Adjust num_heads so that hidden_dim / num_heads is an integer."
            )

        # batch_first=True keeps tensor layout consistent with the LSTM output
        # (batch, seq, dim) without requiring an explicit transpose.
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: LSTM output of shape [Batch, Seq_Len, Hidden_Dim].

        Returns:
            context (torch.Tensor):
                Mean-pooled context vector of shape [Batch, Hidden_Dim].
                Passed directly to the classifier head.
            attn_weights (torch.Tensor):
                Attention weight matrix of shape [Batch, Seq_Len, Seq_Len],
                averaged over all heads. Use for heatmap visualisation.
        """
        # Self-attention: query = key = value = x
        # need_weights=True + average_attn_weights=True are set explicitly so
        # that PyTorch never silently falls back to the fast-path that returns
        # attn_weights=None (observed in torch >= 2.0 under certain conditions).
        # attn_output  : [Batch, Seq_Len, Hidden_Dim]
        # attn_weights : [Batch, Seq_Len, Seq_Len]  (avg over heads)
        attn_output, attn_weights = self.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=True,
        )

        # Mean-pool attended sequence -> single context vector per sample.
        # Mean pooling is preferred over last-step selection because all
        # time steps in a network flow can carry attack-relevant information.
        context = attn_output.mean(dim=1)   # [Batch, Hidden_Dim]

        return context, attn_weights


# ---------------------------------------------------------------------------
# Backward-compatible alias
# Existing imports of the form `from src.models.attention import SelfAttention`
# continue to resolve without changes in specialist model files.
# ---------------------------------------------------------------------------
SelfAttention = MultiHeadSelfAttention
