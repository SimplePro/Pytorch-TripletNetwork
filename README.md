## TripletNetwork
- https://arxiv.org/pdf/1412.6622.pdf


### Model Structure
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 24, 24]             832
         MaxPool2d-2           [-1, 32, 12, 12]               0
              ReLU-3           [-1, 32, 12, 12]               0
            Conv2d-4           [-1, 64, 10, 10]          18,496
         MaxPool2d-5             [-1, 64, 5, 5]               0
              ReLU-6             [-1, 64, 5, 5]               0
            Conv2d-7            [-1, 128, 4, 4]          32,896
         MaxPool2d-8            [-1, 128, 2, 2]               0
              ReLU-9            [-1, 128, 2, 2]               0
           Conv2d-10            [-1, 128, 1, 1]          65,664
          Dropout-11            [-1, 128, 1, 1]               0
           Conv2d-12           [-1, 32, 24, 24]             832
        MaxPool2d-13           [-1, 32, 12, 12]               0
             ReLU-14           [-1, 32, 12, 12]               0
           Conv2d-15           [-1, 64, 10, 10]          18,496
        MaxPool2d-16             [-1, 64, 5, 5]               0
             ReLU-17             [-1, 64, 5, 5]               0
           Conv2d-18            [-1, 128, 4, 4]          32,896
        MaxPool2d-19            [-1, 128, 2, 2]               0
             ReLU-20            [-1, 128, 2, 2]               0
           Conv2d-21            [-1, 128, 1, 1]          65,664
          Dropout-22            [-1, 128, 1, 1]               0
           Conv2d-23           [-1, 32, 24, 24]             832
        MaxPool2d-24           [-1, 32, 12, 12]               0
             ReLU-25           [-1, 32, 12, 12]               0
           Conv2d-26           [-1, 64, 10, 10]          18,496
        MaxPool2d-27             [-1, 64, 5, 5]               0
             ReLU-28             [-1, 64, 5, 5]               0
           Conv2d-29            [-1, 128, 4, 4]          32,896
        MaxPool2d-30            [-1, 128, 2, 2]               0
             ReLU-31            [-1, 128, 2, 2]               0
           Conv2d-32            [-1, 128, 1, 1]          65,664
          Dropout-33            [-1, 128, 1, 1]               0
================================================================
Total params: 353,664
Trainable params: 353,664
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1838.27
Forward/backward pass size (MB): 0.93
Params size (MB): 1.35
Estimated Total Size (MB): 1840.54
```

#### Feature Representation Space (by TripletNet)
![epoch_29](https://github.com/SimplePro/Pytorch-TripletNetwork/assets/66504341/ad56bfe6-db99-447a-915e-207863c64d7a)
<div align="center"> TripletNet's 2d feature representation space (epoch29) </div>
</br>

-----------------------------
### Comparing with AutoEncoder's embedding space
#### Feature Representation Space (by AutoEncoder)
- Seeing this result, I became to more exactly understand tripletnet's purpose.    

![epoch_29](https://github.com/SimplePro/Pytorch-TripletNetwork/assets/66504341/5332047e-8e98-4882-aebd-56a494c581b1)
<div align="center"> AutoEncoder's 2d feature representation space (epoch 29) </div>

<br/>
<br/>

<p align="center">
<img aling="center" src="https://github.com/SimplePro/Pytorch-TripletNetwork/assets/66504341/88040b97-0007-44ed-a349-2897b3c6c448">
</p>

<div align="center"> reconstruction images </div>