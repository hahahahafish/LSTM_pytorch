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
#print(model)

# 評估訓練集上的性能(loss)
def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)   # 計算預測輸出跟真實標籤之間的損失
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()     # 根據損失值反向運算，計算梯度
        optimizer.step()    # 使用優化器更新參數，最小化loss

        # 每隔100批次打印平均loss值
        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0     # reset loss
    print()

# 評估驗證集上的性能(loss)
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
        
    avg_loss_across_batches = running_loss / len(test_loader)
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('****************************************************')
    print()

learning_rate = 0.001
num_epoch = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    train_one_epoch()
    validate_one_epoch()

# 用訓練好的模型對"訓練集"上的數據進行預測(為了檢測擬合狀態)
with torch.no_grad():   # 告訴pytorch在接下來的運算中不需要計算梯度
    predicted = model(x_train.to(device)).to('cpu').numpy()    # numpy只能使用cpu不會使用gpu

'''
plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
'''

# 訓練集
# 將-1,1的值轉換為實際美元
train_predictions = predicted.flatten()    # 二維變一維(使用模型的預測值)

dummies = np.zeros((x_train.shape[0], lookback+1))     # 整個陣列的長度(x+y)
dummies[:, 0] = train_predictions   # 第一行為預測值
dummies = scaler.inverse_transform(dummies)    #反轉化(將標準化的數值還原到原始)

train_predictions = dc(dummies[:, 0])
#print(train_predictions)

dummies = np.zeros((x_train.shape[0], lookback+1))     # 整個陣列的長度(x+y)
dummies[:, 0] = y_train.flatten()   # 第一行為預測值，二維變一維(使用實際值)
dummies = scaler.inverse_transform(dummies)    #反轉化(將標準化的數值還原到原始)

new_y_train = dc(dummies[:, 0])
#print(new_y_train)
'''
'''
plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# 測試集
test_predictions = model(x_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((x_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
print(test_predictions)

dummies = np.zeros((x_test.shape[0], lookback+1))     # 整個陣列的長度(x+y)
dummies[:, 0] = y_test.flatten()   # 第一行為預測值，二維變一維(使用實際值)
dummies = scaler.inverse_transform(dummies)    #反轉化(將標準化的數值還原到原始)

new_y_test = dc(dummies[:, 0])
print(new_y_test)

plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()