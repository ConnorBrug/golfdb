import torch
import torch.nn as nn
import timm


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        # MobileNetV3-Large backbone via timm
        self.cnn = timm.create_model('mobilenetv3_large_100', pretrained=pretrain,
                                     num_classes=0, global_pool='avg', drop_rate=0.0)

        self.rnn = nn.LSTM(1280, self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)

        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size, device):
        num_dirs = 2 if self.bidirectional else 1
        h = torch.zeros(num_dirs * self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(num_dirs * self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size, x.device)

        # flatten batch and time dims for the CNN
        c_in = x.view(batch_size * timesteps, C, H, W)
        # channels-last format on the rank-4 tensor, speeds up depthwise convs on Ampere+
        c_in = c_in.contiguous(memory_format=torch.channels_last)
        c_out = self.cnn(c_in)

        if self.dropout:
            c_out = self.drop(c_out)

        # back to rank 3 for the LSTM
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)
        return out