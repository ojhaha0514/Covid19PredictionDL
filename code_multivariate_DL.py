import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import datetime
import warnings
from torch.nn.utils import weight_norm

warnings.simplefilter(action='ignore', category=FutureWarning)
torch.manual_seed(0)
np.random.seed(0)

input_window = 14  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
pred_length = 28
repeat = 1
perms = 5

block_len = input_window + output_window  # for one input-output pair
batch_size = 8
lr = 0.00013030546021945992
epochs = 300
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = datetime.datetime(2022, 1, 1)

nums_country = np.array([0, 1, 3, 5, 10, 20, 30, 45, 60, 90])

ends = np.array([datetime.datetime(2022, 7, 31),
                 datetime.datetime(2022, 8, 31),
                 datetime.datetime(2022, 9, 30),
                 datetime.datetime(2022, 10, 31),
                 datetime.datetime(2022, 11, 30)])

iso_codes = np.array(["KOR", "JPN", "RUS", "ITA", "USA"])


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


def train(train_data, model, optimizer):
    model.train()  # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

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

        total_loss += loss.item()

    return total_loss


# predict the next n steps based on the input data

def get_train_loss(eval_model, data_all, future_data, train_data, val_data, steps, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    data1, _ = get_batch(train_data, 0, 1)

    with torch.no_grad():
        for i in range(0, (len(data_all) - input_window) // output_window):
            # output = eval_model(data_all_tensor[i*output_window:i*output_window+input_window])
            data2, _ = get_batch(train_data, i, 1)
            output = eval_model(data2)
            # (seq-len , batch-size , features-num)
            # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
            data1 = torch.cat((data1, output[-output_window:]))  # [m,m+1,..., m+n+1]
    data1 = data1.reshape(data1.shape[0], data1.shape[2])
    data1 = data1.cpu().detach().numpy()

    train_loss = np.mean(
        (data1[-data_all.shape[0] + input_window:, 0] - data_all[input_window:, 0]) ** 2)
    return train_loss


def get_test_loss(eval_model, data_all, future_scaled, train_data, val_data, steps, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    data, _ = get_batch(val_data, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data)
            # (seq-len , batch-size , features-num)
            # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
            data = torch.cat((data, output[-output_window:]))  # [m,m+1,..., m+n+1]
    data = data.reshape(data.shape[0], data.shape[2])
    data = data.cpu().detach().numpy()

    train_loss = np.mean((data[-pred_length * output_window:, 0] - future_scaled) ** 2)
    return train_loss


def predict_future(eval_model, data_all, future_data, train_data, val_data, steps, epoch, scaler, end):
    eval_model.eval()
    data, _ = get_batch(val_data, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):
            output = eval_model(data)
            # (seq-len , batch-size , features-num)
            # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
            data = torch.cat((data, output[-output_window:]))  # [m,m+1,..., m+n+1]
    data = data.reshape(data.shape[0], data.shape[2])
    data = data.cpu().detach().numpy()
    data = scaler.inverse_transform(data)
    data = np.exp(data)

    data_all = scaler.inverse_transform(data_all)
    data_all = np.exp(data_all)

    data1, _ = get_batch(train_data, 0, 1)

    with torch.no_grad():
        for i in range(0, (len(data_all) - input_window) // output_window):
            data2, _ = get_batch(train_data, i, 1)
            output = eval_model(data2)
            # (seq-len , batch-size , features-num)
            # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
            data1 = torch.cat((data1, output[-output_window:]))  # [m,m+1,..., m+n+1]
    data1 = data1.reshape(data1.shape[0], data1.shape[2])
    data1 = data1.cpu().detach().numpy()
    data1 = scaler.inverse_transform(data1)
    data1 = np.exp(data1)

    # I used this plot to visualize if the model pics up any long therm structure within the data.
    pyplot.figure(figsize=(10, 5))

    pyplot.plot(pd.date_range(start + datetime.timedelta(days=input_window),
                              end + datetime.timedelta(days=output_window * pred_length)),
                [*data1[-data_all.shape[0] + input_window:, 0].tolist(),
                 *data[-output_window * pred_length:, 0].tolist()], color="red", label="predict")
    # [*output_fin.tolist(), *data[-output_window * pred_length:].tolist()], color="red", label="predict")
    pyplot.plot(pd.date_range(start, end + datetime.timedelta(days=output_window * pred_length)),
                [*data_all[:, 0].tolist(), *future_data.reshape(-1, 1)[:, 0].tolist()], color="blue",
                label="real")

    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.title("Daily Cases Prediction in Korea using Transformer model")
    pyplot.legend()
    pyplot.show()
    pyplot.close()

    pred = [*data1[-len(data_all) + input_window - 1:-1, ].tolist(),
            *data[-output_window * pred_length:, ].tolist()]

    return pred


def main():
    for iso_code in iso_codes:
        for end in ends:
            for num in nums_country:
                tr_final = []
                for perm in range(perms):
                    model_type = str(num)
                    series0 = read_csv('/home/jhoh/DL/owid-covid-data_231216_final.csv')
                    # preprocessed COVID-19 cases file
                    series0["Date"] = pd.to_datetime(series0["Date"])
                    series0 = series0[series0["Date"] >= start]
                    series0 = series0[series0["Date"] <= end + pd.Timedelta(days=pred_length * output_window)]
                    series0 = series0[series0["population"] >= 10000000]
                    series1 = series0[[iso_code in c for c in list(series0['iso_code'])]]
                    series1 = series1[["Date", "new_cases_corrected"]]

                    iso_rank = read_csv("/home/jhoh/DL/DTW_rank_220731_final.csv", header=1)
                    # DTW distance rank file
                    iso_code_list = iso_rank[iso_code][:91]

                    iso_code_list = iso_code_list[:num]
                    index_iso = np.argwhere(iso_code_list == iso_code)
                    iso_code_list = np.delete(iso_code_list, index_iso)
                    if perm != 0:
                        np.random.shuffle(iso_code_list)
                    print(iso_code_list)

                    num_country = len(iso_code_list)

                    for i in range(0, num_country):
                        series02 = series0[[iso_code_list[i] in c for c in list(series0['iso_code'])]]
                        series02 = series02[["Date", "new_cases_corrected"]]
                        series02.columns = ["Date", iso_code_list[i]]
                        series1 = pd.merge(series1, series02, on="Date", how="inner")
                    series0 = series1.drop("Date", axis=1)
                    series0 = series0.replace(0, np.NaN)
                    series0 = series0.interpolate(limit_direction="both")

                    series = series0[:int((end - start) / pd.Timedelta(days=1)) + 1]
                    series = series.replace(0, 1)
                    series = np.log(series).to_numpy()

                    scaler = MinMaxScaler()
                    confirmed = scaler.fit_transform(series)

                    future_scaled = scaler.transform(
                        np.log(series0[int((end - start) / pd.Timedelta(days=1)) + 1:int(
                            (end - start) / pd.Timedelta(days=1)) + 1 + pred_length * output_window]).to_numpy())[:,
                                    0]

                    future_data = series0[int((end - start) / pd.Timedelta(days=1)) + 1:int(
                        (end - start) / pd.Timedelta(days=1)) + 1 + pred_length * output_window].iloc[:,
                                  0].to_numpy()

                    train_data = confirmed
                    test_data = confirmed

                    # convert our train and test data into a pytorch train tensor
                    train_sequence = create_inout_sequences(train_data, input_window, output_window)
                    test_sequence = create_inout_sequences(test_data, input_window, output_window)
                    test_sequence = test_sequence[-1:]

                    train_data = train_sequence.to(device)
                    val_data = test_sequence.to(device)

                    model = TransAm(num_country).to(device)
                    # change this part to modify the type of DL model among Transformer, RNN, LSTM, BiLSTM, GRU, CNN-LSTM, TCN

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    best_test_loss = float("inf")
                    best_train_loss = float("inf")
                    loss_train = []
                    loss_test = []

                    for epoch in range(1, epochs + 1):
                        epoch_start_time = time.time()
                        train(train_data, model, optimizer)

                        train_loss = get_train_loss(model, confirmed, future_data, train_data, val_data,
                                                    pred_length, epoch, scaler)
                        loss_train.append(train_loss)

                        test_loss = get_test_loss(model, confirmed, future_scaled, train_data, val_data,
                                                  pred_length, epoch, scaler)
                        loss_test.append(test_loss)

                        if (train_loss < best_train_loss):
                            best_train_loss = train_loss
                            tr_best = predict_future(model, confirmed, future_data, train_data, val_data,
                                                     pred_length, epoch, scaler, end)

                        if (test_loss < best_test_loss):
                            best_test_loss = test_loss

                        tr_list = list()
                        if (epoch % 100 == 0):
                            val_loss = 0
                            tr_b = predict_future(model, confirmed, future_data, train_data, val_data, pred_length,
                                                  epoch, scaler, end)
                            tr_list.append(tr_b)

                        else:
                            val_loss = 0

                        if (epoch % 100 == 0):
                            print('-' * 89)
                            print(
                                '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(
                                    epoch, (time.time() - epoch_start_time),
                                    val_loss, math.exp(val_loss)))
                            print('-' * 89)

                    tr_list.append(np.array(tr_best))
                    tr_final.append(tr_list[-1])

                if num == 0 or num == 1:
                    break

                prep = read_csv("/home/jhoh/DL/owid-covid-data_231216_final.csv")
                prep = prep[[iso_code in c for c in list(prep['iso_code'])]]
                prep["Date"] = pd.to_datetime(prep["Date"])
                prep = prep[prep["Date"] >= start]
                prep = prep[["new_cases_corrected"]]
                prep = prep.to_numpy()
                prep = prep[:int((end - start) / pd.Timedelta(days=1)) + output_window * pred_length + 1]

                csv1 = pd.DataFrame(
                    {"Date": list(pd.date_range(start, end + datetime.timedelta(days=output_window * pred_length))),
                     "Actual": list(prep)})

                if num_country == 0:
                    csv2 = pd.DataFrame(np.vstack([np.full(shape=(input_window, num_country + 1), fill_value=None),
                                                   np.mean(tr_final, axis=0).reshape(-1, num_country + 1)[
                                                   -len(prep) + input_window:, ]]))
                    csv2.columns = [iso_code]
                else:
                    csv2 = pd.DataFrame(np.vstack([np.full(shape=(input_window, num_country + 1), fill_value=None),
                                                   np.mean(tr_final, axis=0)[-len(prep) + input_window:, ]]))
                    csv2.columns = np.insert(iso_code_list, 0, iso_code)
                csv = pd.concat([csv1, csv2], axis=1)
                csv = csv.set_index("Date")
                csv.to_csv(
                    "/data/member/jhoh/result/result_multi_perms_" + str(epochs) + "_" + str(iso_code) + "_" + str(
                        model_type) + "_" + str(
                        end) + ".csv")

                del model


main()


