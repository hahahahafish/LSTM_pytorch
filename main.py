import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

data = pd.read_csv('AMZN.csv')


data = data[['Date', 'Close']]

#print(data)

data['Date'] = pd.to_datetime(data['Date'])
#plt.plot(data['Date'], data['Close'])
#plt.show()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#print(device)


def prepare_datafrme_for_lstm(df, n_steps):
    # 不要動到Dataframe裡原本的data，所以採用副本方式進行
    df = dc(df)
    df['Date'] = pd.to_datetime(df['Date'])

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    return df

# 考慮過去i天的收盤價
lookback = 7
shifted_df = prepare_datafrme_for_lstm(data, lookback)
#print(shifted_df)

shifted_df_as_np = shifted_df.to_numpy()
#print(shifted_df_as_np)

# 將數據映射到統一的範圍
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
#print(shifted_df_as_np)

# 提取特徵
x = shifted_df_as_np[:, 1:] # 所有列、第二行到最後一行
y = shifted_df_as_np[:, 0]  # 所有列、第一行
#print(x.shape, y.shape)

#x = dc(np.flip(x, axis=1))

# 切割訓練集
split_index = int(len(x) * 0.95)
#print(split_index)

x_train = x[:split_index]
x_test = x[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 將訓練集跟測試集擴展成三維
x_train = x_train.reshape((-1, lookback, 1))
x_test = x_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 將資料及轉換為pytorch張量，方便與pytorch框架兼容
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 定義時間序資料集
class TimeSeriesDataset(Dataset):
    # 初始化x和y
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # 樣本數
    def __len__(self):
        return len(self.x)
    
    # 第i個樣本的x和y
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
# 將資料集載入TimeSeriesDataset
train_dataset = TimeSeriesDataset(x_train,y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 驗證loader是否正常運行
for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    #print(x_batch.shape, y_batch.shape)
    break

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
# LSTM維度:input=1，hidden=4，stacked=1
model = LSTM(1, 4, 1)
model.to(device)
print(model)

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    rinning_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()


learning_rate = 0.001
num_epoch = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    train_one_epoch()
    validate_one_epoch()