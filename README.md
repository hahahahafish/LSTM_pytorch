# LSTM_pytorch
https://www.notion.so/LSTM-4a992c1f15d04f018a177bc09a298ecd
# 目標

針對Amazon股票，用過去七天的收盤價(x, 第二行到最後一行) 預測今天的收盤價(y, 第一行)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e127742d-72b6-414d-93a4-1c69d86e2b69/263f4878-5032-45ea-90b9-cf4a331e5526/Untitled.png)

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

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e127742d-72b6-414d-93a4-1c69d86e2b69/320d1321-8663-499e-9e2c-9b2eef0b1b84/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e127742d-72b6-414d-93a4-1c69d86e2b69/94c2c59a-d788-4e70-9193-e802a125b828/Untitled.png)

# 模型擬合狀態

## 訓練集未轉換

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e127742d-72b6-414d-93a4-1c69d86e2b69/69eb3664-c36c-41d9-b121-0f3bdd876ed6/Untitled.png)

## 訓練集已轉換

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e127742d-72b6-414d-93a4-1c69d86e2b69/456dea6f-baea-4eed-8e4a-d1d358c869d2/Untitled.png)

# 測試集預測結果
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/e127742d-72b6-414d-93a4-1c69d86e2b69/63b96ae5-f2ce-461f-8a71-7e3da097031b/Untitled.png)
