## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        ## Image size 224 x 224 x 1
        
        ## After conv becomes 220 x 220 x 32 and after pool becomes 110 x 110 x 32
        self.conv1 = nn.Conv2d(1,32,5)
       
        self.pool= nn.MaxPool2d(2,2)
        
        ## After conv becomes 108 x 108 x 64 and after pool becomes 54 x 54 x 64
        self.conv2 = nn.Conv2d(32,64,3)
        
        ##After conv becomes 52 x 52 x 128 and after pool becomes 26 x 26 x 128
        self.conv3 = nn.Conv2d(64,128,3)
   
        ##After conv becomes 26 x 26 x 256 and after pool becomes 13 x 13 x 256
        self.conv4 = nn.Conv2d(128,256,1)
        
        self.fc1 = nn.Linear(256*13*13,1000)
        self.fc1_drop = nn.Dropout(p=0.3)
       
        self.fc2 = nn.Linear(1000,500)
        self.fc2_drop = nn.Dropout(p=0.4)
        
        self.fc3 = nn.Linear(500,136)
        
        
        
      
       
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
       
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0),-1)

        x = self.fc1_drop(F.relu(self.fc1(x)))
        x = self.fc2_drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
