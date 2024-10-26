# UNet
U-Net is a convolutional neural network (CNN) that's used for **image segmentation**, particularly in biomedical applications. 

#### import

```python
from atanu_mlkit.models import UNET
```

```python
in_channels = 3 # Number of channels
num_classes = 5 # Number of classes

model = UNet(in_channels, num_classes)
```