import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import time
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import datetime
import warnings
import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from torch.nn.utils import weight_norm

warnings.simplefilter(action='ignore', category=FutureWarning)

input_window = 14  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
pred_length = 14
repeat = 1

block_len = input_window + output_window  # for one input-output pair
epochs = 300
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = datetime.datetime(2022, 1, 1)

num = 0

end= datetime.datetime(2022, 8, 1)

iso_code = "KOR"


torch.manual_seed(0)
np.random.seed(0)
model_type = str(num)
series0 = read_csv('/home/jhoh/DL/owid-covid-data_231216_final.csv')
# series0 = read_csv('/content/drive/MyDrive/Colab Notebooks/owid-covid-data_231216.csv')
series0["Date"] = pd.to_datetime(series0["Date"])
series0 = series0[series0["Date"] >= start]
series0 = series0[series0["Date"] <= end + pd.Timedelta(days=pred_length * output_window)]
series0 = series0[series0["population"] >= 10000000]
series1 = series0[[iso_code in c for c in list(series0['iso_code'])]]
series1 = series1[["Date", "new_cases_corrected"]]

iso_rank = read_csv("/home/jhoh/DL/DTW_rank_220731.csv", header=1)
iso_code_list = iso_rank[iso_code][:91]

iso_code_list = iso_code_list[:num]

num_country = len(iso_code_list)

for i in range(0, num_country):
    series02 = series0[[iso_code_list[i] in c for c in list(series0['iso_code'])]]
    series02 = series02[["Date", "new_cases_corrected"]]
    series02.columns = ["Date", iso_code_list[i]]
    series1 = pd.merge(series1, series02, on="Date", how="inner")
series0 = series1.drop("Date", axis=1)
series0 = series0.replace(0, np.NaN)
series0 = series0.interpolate(limit_direction="both")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1)  # [5000, 1, d_model],so need seq-len <= 5000
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1, x.shape[1], 1)


class TransAm(nn.Module):
    def __init__(self, num_country, feature_size=36, num_layers=3, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding = nn.Linear(num_country + 1, feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, num_country + 1)
        # self.decoder2 = nn.Linear(2*num_country+1, num_country+1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder2.bias.data.zero_()
        # self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src with shape (input_window, batch_len, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.input_embedding(src)  # linear transformation before positional embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        # output = F.relu(output)
        # output = self.decoder2(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class RNNModel(nn.Module):
    def __init__(self, num_country, hidden_size=16, num_layers=4):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(num_country + 1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_country + 1)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self,num_country, hidden_size=16, num_layers=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_country + 1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_country + 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self,num_country, hidden_size=16, num_layers=4):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_country + 1, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_country + 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self,num_country, hidden_size=16, num_layers=4):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(num_country + 1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_country + 1)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self,num_country, hidden_size=16, num_layers=2):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_country + 1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1)
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_country + 1)

    def forward(self, x):
        # Assuming input shape (seq_len, batch_size, input_dim)
        x = x.permute(1, 2, 0)  # Convert to (batch_size, input_dim, seq_len) for Conv1d
        x = self.cnn(x)
        x = x.permute(2, 0, 1)  # Convert back to (seq_len, batch_size, num_features)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                      dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self,num_country, num_channels=[25, 50, 100], kernel_size=2):
        num_inputs = num_country + 1
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     padding=(kernel_size - 1) * dilation_size, dilation=dilation_size)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_country + 1)

    def forward(self, x):
        x = x.permute(1, 2, 0)  # Convert to (batch_size, input_dim, seq_len)
        y = self.network(x)
        y = y.permute(2, 0, 1)  # Convert back to (batch_size, seq_len, num_channels[-1])
        out = self.fc(y)
        return out


def create_inout_sequences(input_data, input_window, output_window):
    inout_seq = []
    L = len(input_data)
    block_num = L - block_len + 1
    # total of [N - block_len + 1] blocks
    # where block_len = input_window + output_window

    for i in range(block_num):
        train_seq = input_data[i: i + input_window]
        train_label = input_data[i + output_window: i + input_window + output_window]
        # train_label[:input_window-output_window]=0
        inout_seq.append((train_seq, train_label))

    return torch.FloatTensor(np.array(inout_seq))


def get_batch(input_data, i, batch_size):

    # batch_len = min(batch_size, len(input_data) - 1 - i) #  # Now len-1 is not necessary
    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[i:i + batch_len]

    if data.ndim >= 4:
        shape = data.shape[3]
    else:
        shape = 1

    input = torch.stack([item[0] for item in data]).view((input_window, -1, shape))
    # input = torch.stack([item[0] for item in data]).view((input_window,batch_len,1))
    # ( seq_len, batch, 1 ) , 1 is feature size

    target = torch.stack([item[1] for item in data]).view((input_window, -1, shape))
    return input, target


def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = TransAm(config["feature_size"], config["num_layers"], config["drop_out"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)


    for epoch in range(300):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        net.train()  # Turn on the train mode \o/

        series = series0[:int((end - start) / pd.Timedelta(days=1)) + 1]
        series = series.replace(0, 1)
        series = np.log(series).to_numpy()

        scaler = MinMaxScaler()
        confirmed = scaler.fit_transform(series)
        train_data = confirmed
        test_data = confirmed
        train_sequence = create_inout_sequences(train_data, input_window, output_window)
        test_sequence = create_inout_sequences(test_data, input_window, output_window)
        test_sequence = test_sequence[-1:]  # todo: fix hack?

        train_data = train_sequence.to(device)
        val_data = test_sequence.to(device)

        batch_size = config["batch_size"]

        for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
            # data and target are the same shape with (input_window,batch_len,1)
            data, targets = get_batch(train_data, i, batch_size)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            # loss = abs(loss - b) + b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0



        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        for batch, i in enumerate(range(0, len(val_data), batch_size)):  # Now len-1 is not necessary
            # data and target are the same shape with (input_window,batch_len,1)
            with torch.no_grad():
                data, targets = get_batch(val_data, i, batch_size)
                output = model(data)
                loss = criterion(output, targets)
                val_loss += loss.cpu().numpy()
                val_steps += 1


        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)

        ray.train.report(dict(loss=val_loss))
    print("Finished Training")

model = TransAm().to(device)
# change this part to modify the type of DL model among Transformer, RNN, LSTM, BiLSTM, GRU, CNN-LSTM, TCN

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "feature_size": tune.sample_from(lambda _: 4 * np.random.randint(4, 17)),
        "num_layers": tune.sample_from(lambda _: np.random.randint(1, 5)),
        "drop_out": tune.choice([0, 0.1, 0.2, 0.3]),
        "lr": tune.loguniform(1e-5, 2*1e-4),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        #metric_columns=["loss", "training_iteration"])
        metric_columns=["loss"])
    result = tune.run(
        partial(train_cifar),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=2000, max_num_epochs=1000, gpus_per_trial=2)