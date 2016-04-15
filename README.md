# adversarial_autoencoders

# 準備
MNISTのデータを用意する

```
./data/mnist/t10k-images-idx3-ubyte.gz
./data/mnist/t10k-labels-idx1-ubyte.gz
./data/mnist/train-images-idx3-ubyte.gz
./data/mnist/train-labels-idx1-ubyte.gz
```
# 実行
python adversarial_autoencoders.py

# 結果
## 10 2D gaussian

![](./example/gaussian/q_z_f10.gif)

||||
|---|---|---|
|![](./example/gaussian/genimg_d00.gif)|![](./example/gaussian/genimg_d01.gif)|![](./example/gaussian/genimg_d02.gif)|
|![](./example/gaussian/genimg_d03.gif)|![](./example/gaussian/genimg_d04.gif)|![](./example/gaussian/genimg_d05.gif)|
|![](./example/gaussian/genimg_d06.gif)|![](./example/gaussian/genimg_d07.gif)|![](./example/gaussian/genimg_d08.gif)|
|![](./example/gaussian/genimg_d09.gif)|||

## Swiss Roll

![](./example/swissroll/q_z_f10.gif)

||||
|---|---|---|
|![](./example/swissroll/genimg_d00.gif)|![](./example/swissroll/genimg_d01.gif)|![](./example/swissroll/genimg_d02.gif)|
|![](./example/swissroll/genimg_d03.gif)|![](./example/swissroll/genimg_d04.gif)|![](./example/swissroll/genimg_d05.gif)|
|![](./example/swissroll/genimg_d06.gif)|![](./example/swissroll/genimg_d07.gif)|![](./example/swissroll/genimg_d08.gif)|
|![](./example/swissroll/genimg_d09.gif)|||
