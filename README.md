CUDA-NN
======================
CUDA program for classifying MNIST-like images using a feed-forward network

Usage
-----
`.\cuda-nn.exe`

Output
------
```
Using GeForce GTX 1060
[0 ms] Parsing ../dat/fashion/train-images-idx3-ubyte.gz
[220 ms] Parsing ../dat/fashion/train-labels-idx1-ubyte.gz
[224 ms] Parsing ../dat/fashion/t10k-images-idx3-ubyte.gz
[263 ms] Parsing ../dat/fashion/t10k-labels-idx1-ubyte.gz
[266 ms] Uploading train images @0[6000]
[305 ms] Uploading train images @6000[6000]
[344 ms] Uploading train images @12000[6000]
[381 ms] Uploading train images @18000[6000]
[420 ms] Uploading train images @24000[6000]
[459 ms] Uploading train images @30000[6000]
[496 ms] Uploading train images @36000[6000]
[534 ms] Uploading train images @42000[6000]
[572 ms] Uploading train images @48000[6000]
[611 ms] Uploading train images @54000[6000]
[648 ms] Uploading test images
[764 ms] Epoch 1, train 10.0%, test 10.0%
[19887 ms] Epoch 500, train 88.4%, test 85.9%
[38978 ms] Epoch 1000, train 90.6%, test 86.9%
[58077 ms] Epoch 1500, train 91.2%, test 86.7%
[77180 ms] Epoch 2000, train 91.7%, test 87.0%
[96289 ms] Epoch 2500, train 91.2%, test 86.5%
[115565 ms] Epoch 3000, train 92.1%, test 86.8%
[134767 ms] Epoch 3500, train 93.7%, test 86.8%
[153945 ms] Epoch 4000, train 93.5%, test 86.9%
[173138 ms] Epoch 4500, train 93.4%, test 86.6%
[192323 ms] Epoch 5000, train 94.7%, test 87.2%
[211502 ms] Epoch 5500, train 93.5%, test 85.9%
[230674 ms] Epoch 6000, train 95.4%, test 86.7%
[249864 ms] Epoch 6500, train 95.1%, test 86.6%
[269035 ms] Epoch 7000, train 94.8%, test 86.3%
[288216 ms] Epoch 7500, train 94.7%, test 85.9%
[307379 ms] Epoch 8000, train 95.8%, test 86.2%
[326556 ms] Epoch 8500, train 95.9%, test 86.3%
[345745 ms] Epoch 9000, train 95.3%, test 85.9%
[364941 ms] Epoch 9500, train 96.0%, test 86.0%
[384101 ms] Epoch 10000, train 96.3%, test 85.6%
```
