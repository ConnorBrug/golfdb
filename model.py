"""Event detection model: MobileNetV3-Large CNN per frame + BiLSTM over time +
linear head for 9 classes (8 swing events + no-event).

Input: (B, T, 3, 160, 160) float tensor (normalized upstream in gpu_augment.py).
Output: (B*T, 9) class logits.
"""

import torch
import torch.nn as nn
import timm


class EventDetector(nn.Module):
    def __init__(
        self,
        pretrain: bool,
        width_mult: float,
        lstm_layers: int,
        lstm_hidden: int,
        bidirectional: bool = True,
        dropout: bool = False,
        cnn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        lstm_dropout: float = 0.0,
        checkpoint_backbone: bool = False,
    ):
        super().__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.cnn = timm.create_model(
            'mobilenetv3_large_100',
            pretrained=pretrain,
            num_classes=0,
            global_pool='avg',
            drop_rate=cnn_dropout,
            drop_path_rate=drop_path_rate,
        )

        if checkpoint_backbone and hasattr(self.cnn, 'set_grad_checkpointing'):
            self.cnn.set_grad_checkpointing(enable=True)

        # timm 1.0+ renamed the post-conv_head dim to head_hidden_size; num_features is now the
        # pre-head block dim (960 for mobilenetv3_large_100, wrong for our use). Prefer
        # head_hidden_size, fall back to num_features for older timm, final fallback is a probe.
        feat_dim = getattr(self.cnn, 'head_hidden_size', None)
        if feat_dim is None:
            feat_dim = self.cnn.num_features
        # sanity: run a tiny forward to confirm the LSTM input_size matches the backbone output
        with torch.no_grad():
            was_training = self.cnn.training
            self.cnn.eval()
            probe = self.cnn(torch.zeros(1, 3, 160, 160))
            actual = probe.shape[-1]
            if was_training:
                self.cnn.train()
        if actual != feat_dim:
            feat_dim = actual

        # LSTM dropout is applied between stacked layers only, so it needs layers>1 to have any effect
        self.rnn = nn.LSTM(
            input_size=feat_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if self.lstm_layers > 1 else 0.0,
        )

        head_in = 2 * self.lstm_hidden if self.bidirectional else self.lstm_hidden
        self.lin = nn.Linear(head_in, 9)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = x.shape

        # flatten time into batch, then switch to channels_last for the conv backbone
        c_in = x.reshape(batch_size * timesteps, channels, height, width)
        c_in = c_in.contiguous(memory_format=torch.channels_last)

        c_out = self.cnn(c_in)

        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM always runs in fp32 for stability, None hidden defaults to zeros and saves an alloc
        r_in = c_out.reshape(batch_size, timesteps, -1).float()
        r_out, _ = self.rnn(r_in)
        out = self.lin(r_out)
        return out.reshape(batch_size * timesteps, 9)
