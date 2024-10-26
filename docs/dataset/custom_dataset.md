# Custom Dataset


## Dataset
```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
```

## DataSplit 

```python
import torch.utils.data as data

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))   
test_size = len(dataset) - train_size - val_size 

train_data, val_data, test_data = data.random_split(
    dataset, 
    [train_size, val_size, test_size])

```

## Dataloader 
```python

from torch.utils.data import Dataset, DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
```