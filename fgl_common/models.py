"""FGL 共享模型库。

所有模型接受形状 (batch, 1, lookback_window) 的输入。

- ``RNN``         —— 2 层 RNN + FC-ReLU-FC,分类头(输出 num_bins 个 logits)。
                     teacher/student 的默认模型。迁自 ``mackey_glass/utils/utils.py``。
- ``LSTMModel``   —— 2 层 LSTM 分类对照。收自 ``cstr/exp/fgl_cstr_lstm.py``。
- ``RNNRegression``—— RNN 连续值回归(输出标量)。收自 ``cstr/exp/fgl_cstr_regression.py``。
- ``SeqRNN``      —— 多步序列预测(输出 ``output_steps × num_bins``)。收自 ``cstr/exp/fgl_cstr_seq2seq.py``。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """2-layer RNN with dropout and FC-ReLU-FC classification head."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0),
                         self.rnn.hidden_size).to(x.device)
        out, hidden = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out


class LSTMModel(nn.Module):
    """2-layer LSTM, same hidden size / structure as ``RNN`` (classification)."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, 1, lookback_window)
        out, _ = self.lstm(x)            # (batch, lookback, hidden)
        out = out[:, -1, :]              # last timestep
        out = F.relu(self.fc1(out))
        out = self.fc2(out)              # (batch, num_bins)
        return out


class RNNRegression(nn.Module):
    """RNN for continuous-value prediction (single scalar output)."""

    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0),
                         self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)
        out = self.fc2(out)
        return out.squeeze(-1)           # (batch,)


class SeqRNN(nn.Module):
    """Encoder RNN → multi-output head predicting ``output_steps`` values,
    each discretized into ``num_bins`` bins.

    Output shape: ``(batch, output_steps, num_bins)``.
    """

    def __init__(self, input_size, hidden_size, output_steps, num_bins, num_layers=2):
        super().__init__()
        self.output_steps = output_steps
        self.num_bins = num_bins
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_steps * num_bins)

    def forward(self, x):
        # x: (batch, 1, lookback_window)
        h0 = torch.zeros(self.rnn.num_layers, x.size(0),
                         self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]              # (batch, hidden_size)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)              # (batch, output_steps * num_bins)
        return out.view(-1, self.output_steps, self.num_bins)
