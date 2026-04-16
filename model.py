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
        checkpoint_backbone: bool = False,
    ):
        super().__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        # timm returns pooled 1280-d features when num_classes=0 and global_pool='avg'
        self.cnn = timm.create_model(
            'mobilenetv3_large_100',
            pretrained=pretrain,
            num_classes=0,
            global_pool='avg',
            drop_rate=cnn_dropout,
        )

        # Enable activation checkpointing when you need lower VRAM at the cost of some speed.
        if checkpoint_backbone and hasattr(self.cnn, 'set_grad_checkpointing'):
            self.cnn.set_grad_checkpointing(enable=True)

        self.rnn = nn.LSTM(
            input_size=1280,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        head_in = 2 * self.lstm_hidden if self.bidirectional else self.lstm_hidden
        self.lin = nn.Linear(head_in, 9)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def init_hidden(self, batch_size: int, device: torch.device):
        num_dirs = 2 if self.bidirectional else 1
        h = torch.zeros(num_dirs * self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(num_dirs * self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns logits: (B*T, 9)
        """
        batch_size, timesteps, channels, height, width = x.shape
        hidden = self.init_hidden(batch_size, x.device)

        # Flatten B and T for the CNN, use channels-last for faster depthwise convs on RTX 40xx.
        c_in = x.reshape(batch_size * timesteps, channels, height, width)
        c_in = c_in.contiguous(memory_format=torch.channels_last)

        c_out = self.cnn(c_in)  # (B*T, 1280)

        if self.dropout:
            c_out = self.drop(c_out)

        r_in = c_out.reshape(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in, hidden)
        out = self.lin(r_out)
        return out.reshape(batch_size * timesteps, 9)
