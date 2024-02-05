# LSTM_pytorch
# 目標

針對Amazon股票，用過去七天的收盤價(x, 第二行到最後一行) 預測今天的收盤價(y, 第一行)

![image](https://github.com/hahahahafish/LSTM_pytorch/assets/151550763/07b3c40c-cae4-4e35-aca4-f480808c2faa)


# 步驟

1. 資料初始化 (映射至統一範圍)
2. 提取特徵
3. 切割資料集 (訓練集95%、測試集5%)
4. 將資料集擴展為三維，轉換型態與pytorch兼容
5. 定義時間序
6. 開始訓練
7. 利用訓練集評估模型擬合狀態
8. 測試結果

# 模型訓練結果

![image](https://github.com/hahahahafish/LSTM_pytorch/assets/151550763/eaa629a8-777d-46bd-acbb-f2bdab8a2b10)


# 模型擬合狀態

## 訓練集未轉換

![image](https://github.com/hahahahafish/LSTM_pytorch/assets/151550763/7958cc5d-c70b-4317-a418-0d71681eb27c)

## 訓練集已轉換

![image](https://github.com/hahahahafish/LSTM_pytorch/assets/151550763/8d3d464b-5aad-4600-a294-5d3557568475)

# 測試集預測結果
![image](https://github.com/hahahahafish/LSTM_pytorch/assets/151550763/3b8ed854-9394-4f3a-b9d4-878ffe24d223)

