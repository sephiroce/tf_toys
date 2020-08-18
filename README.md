# tf_toys
- simplest toy examples using Tensorflow 2 

## Pre-requisite
- ```pip install -r requirements.txt```
- basepath: tf_toys

## Tutorial
- python files will automatically download data from the server.
- cnn_mnist.py 
```
Epoch 1 loss 1.087, accuracy 0.864
Epoch 2 loss 0.270, accuracy 0.919
Epoch 3 loss 0.237, accuracy 0.928
Epoch 4 loss 0.223, accuracy 0.932
Epoch 5 loss 0.206, accuracy 0.937
Epoch 6 loss 0.195, accuracy 0.940
Epoch 7 loss 0.184, accuracy 0.943
Epoch 8 loss 0.171, accuracy 0.947
Epoch 9 loss 0.157, accuracy 0.951
Test loss 0.259, accuracy 0.939
```

- capsnet/dynamic_routing.py
* almost 6 minutes for one epoch.
```
Training the model: 1875/1200 Loss 3.577 Acc 100.000
Epoch: 1 Loss 3.658, Acc 0.957, 221.240 secs elapsed
Epoch: 1  Val accuracy: 98.6022%  Loss: 3.692710 (improved)
Training the model: 1875/1200 Loss 3.462 Acc 96.875
Epoch: 2 Loss 3.623, Acc 0.989, 220.080 secs elapsed
Epoch: 2  Val accuracy: 98.6821%  Loss: 3.686636 (improved)
Training the model: 1875/1200 Loss 3.309 Acc 100.000
Epoch: 3 Loss 3.619, Acc 0.993, 220.154 secs elapsed
Epoch: 3  Val accuracy: 99.0515%  Loss: 3.684260 (improved)
Training the model: 1875/1200 Loss 3.836 Acc 100.000
Epoch: 4 Loss 3.616, Acc 0.995, 220.418 secs elapsed
Epoch: 4  Val accuracy: 99.1514%  Loss: 3.680460 (improved)
Training the model: 1875/1200 Loss 3.705 Acc 100.000
Epoch: 5 Loss 3.615, Acc 0.996, 220.480 secs elapsed
Epoch: 5  Val accuracy: 99.1414%  Loss: 3.682632
Training the model: 1875/1200 Loss 3.884 Acc 100.000
Epoch: 6 Loss 3.614, Acc 0.997, 220.402 secs elapsed
Epoch: 6  Val accuracy: 99.1014%  Loss: 3.680051 (improved)
Training the model: 1875/1200 Loss 3.386 Acc 100.000
Epoch: 7 Loss 3.613, Acc 0.998, 220.084 secs elapsed
Epoch: 7  Val accuracy: 98.9018%  Loss: 3.688487
Training the model: 1875/1200 Loss 3.938 Acc 100.000
Epoch: 8 Loss 3.612, Acc 0.998, 220.295 secs elapsed
Epoch: 8  Val accuracy: 99.2212%  Loss: 3.679805 (improved)
Training the model: 1875/1200 Loss 3.149 Acc 100.000
Epoch: 9 Loss 3.611, Acc 0.999, 220.403 secs elapsed
Epoch: 9  Val accuracy: 98.9617%  Loss: 3.683250
Training the model: 1875/1200 Loss 3.555 Acc 100.000
Epoch: 10 Loss 3.612, Acc 0.999, 220.665 secs elapsed
Epoch: 10  Val accuracy: 99.1514%  Loss: 3.680439
Evaluation!
Final test accuracy: 99.1514%  Loss: 3.680439
```