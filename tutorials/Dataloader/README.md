# PyTorch Dataset and DataLoader

In this notebook, we show an example of how to iteratively access the data in PyTorch style.  
By default, the dataset is one-hot encoded, below we show a how to make a flexible dataloader that provides different label formats.  
Specifically, we show 3 formats:  

- one-hot (default): ideal for classification tasks
- scalar: a single value, useful for regression tasks
- ordinal: a binary ordinal encoding that can be used in an ordinal regression context

 Label encoding table:
 ```
                    Scalar | Ordinal |  One-hot  |  Description
                        0  | 0 0 0 0 | 1 0 0 0 0 |  Very low
                        1  | 1 0 0 0 | 0 1 0 0 0 |  Low
                        2  | 1 1 0 0 | 0 0 1 0 0 |  Average
                        3  | 1 1 1 0 | 0 0 0 1 0 |  High
                        4  | 1 1 1 1 | 0 0 0 0 1 |  Very high
```


```python
import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

label_names = {0:"very low", 1:"low", 2:"average", 3:"high", 4:"very high"}
```


```python

class PotentialDataset(Dataset):
    """
    Multi-temporal-spectral Dataset
    
    Parameters:
        data_path (str): h5 file path of the dataset.
        label_name (str): label name  (potential name: {viticulture, field, market}) 
                        
        mode (str): Specifies the dataset mode. Possible values are:
                    - "train": Use training data
                    - "val": Use validation data
                    - "test": Use test data
                    Default is "train"

        label_format (str): Specifies the format of the labels. Options include:
                    - "scalar": single-channel scalar values
                    - "ordinal": 4-channel ordinal encoding
                    - "one-hot": 5-channel one-hot encoded labels
                    Label encoding table:
                    Scalar | Ordinal |  One-hot  |  Description
                        0  | 0 0 0 0 | 1 0 0 0 0 |  Very low
                        1  | 1 0 0 0 | 0 1 0 0 0 |  Low
                        2  | 1 1 0 0 | 0 0 1 0 0 |  Average
                        3  | 1 1 1 0 | 0 0 0 1 0 |  High
                        4  | 1 1 1 1 | 0 0 0 0 1 |  Very high
                    Default is "one-hot"
    """
    def __init__(self, data_path, label_name, label_format="one-hot",  mode="train"):
        assert mode in ["train", "val", "test"], "mode: specifies the dataset mode. Possible values are: train, val, test."
        self.data_path = data_path
        self.mode = mode
        self.label_format = label_format
        self.label_name = label_name
        
        with h5py.File(self.data_path, 'r') as dataset:
            self.dataset_size = dataset[self.mode]["labels"][self.label_name].shape[0]
            self.timestamps = sorted(list(dataset[self.mode]["sentinel2"].keys()))
            self.n_channels = dataset[self.mode]["sentinel2"][self.timestamps[0]].shape[1]
            self.patch_size = dataset[self.mode]["sentinel2"][self.timestamps[0]].shape[2]
            
    def __len__(self):
        return self.dataset_size
        
    def __getitem__(self, idx):
        """
        Retrieves a time series of 11 Sentinel-2 images and the corresponding labels for a given index

        Returns:
            masks: Tensor of shape (H, W) that represents a binary mask of labelled pixels (1) and ignore labels (0)
            inputs: Tensor of shape (T, C, H, W), where:
                - T is the number of timestamps
                - C is the number of spectral channels
                - H and W are the spatial dimensions
            labels: Tensor of shape (X, H, W) corresponding to the encoding of the ground truth where
                    X depends on self.label_format
        """
        with h5py.File(self.data_path, 'r') as dataset:
            inputs = np.zeros((len(self.timestamps), self.n_channels, self.patch_size, self.patch_size))
            for i in range(len(self.timestamps)):
                inputs[i] = dataset[self.mode]["sentinel2"][self.timestamps[i]][idx]

            labels =  dataset[self.mode]["labels"][self.label_name][idx]
            labels = np.transpose(labels, (2, 0, 1))

            inputs = torch.tensor(inputs)
            labels = torch.tensor(labels)

            inputs = inputs.to(torch.float32)
            masks = torch.sum(labels, 0)

            if self.label_format == "one-hot":
                    pass # By default the dataset has labels encoded in one-hot format
            elif self.label_format == "scalar":
                    labels = torch.argmax(labels, 0).to(torch.float32)
            elif self.label_format ==  "ordinal":
                    class_idx = torch.argmax(labels, 0)
                    num_classes = 5
                    thresholds = torch.arange(1, num_classes)
                    labels = (class_idx >= thresholds.view(-1, 1, 1)).float()
            else:
                 raise ValueError(f"Invalid label encoding format: {self.label_format}")
            
        return masks, inputs, labels
```

## Let's give it a try

### one-hot


```python
onehot_dataset = PotentialDataset(
    data_path = ,# your data path
    label_name = "viticulture",
    label_format = "one-hot",
    mode = "test"
)

onehot_dataloader = DataLoader(onehot_dataset, batch_size=1, shuffle=False)

masks, inputs, labels = next(iter(onehot_dataloader))

print(masks.shape, inputs.shape, labels.shape)
print("Converting one-hot encoded labels to actual class names:")

labels = torch.argmax(labels, axis=1)
print(labels.shape)
print()

print(f"Label value is {int(labels[0,13,13])}, which corresponds to: {label_names[int(labels[0,13,13])]}")

```

    torch.Size([1, 128, 128]) torch.Size([1, 11, 10, 128, 128]) torch.Size([1, 5, 128, 128])
    Converting one-hot encoded labels to actual class names:
    torch.Size([1, 128, 128])
    
    Label value is 4, which corresponds to: very high


### scalar


```python
scalar_dataset = PotentialDataset(
    data_path = "/srv/storage/mgarouanprojects@storage2.nancy.grid5000.fr/moelsakka/agri-potential-dataset/dataset.h5",
    label_name = "viticulture",
    label_format = "scalar",
    mode = "test"
)

scalar_dataloader = DataLoader(scalar_dataset, batch_size=1, shuffle=False)

masks, inputs, labels = next(iter(scalar_dataloader))

print(masks.shape, inputs.shape, labels.shape)
print("Converting scalar encoded labels to actual class names:")
print()

print(f"Label value is {int(labels[0,13,13])}, which corresponds to: {label_names[int(labels[0,13,13])]}")

```

    torch.Size([1, 128, 128]) torch.Size([1, 11, 10, 128, 128]) torch.Size([1, 128, 128])
    Converting scalar encoded labels to actual class names:
    
    Label value is 4, which corresponds to: very high


### Ordinal


```python
ordinal_dataset = PotentialDataset(
    data_path = "/srv/storage/mgarouanprojects@storage2.nancy.grid5000.fr/moelsakka/agri-potential-dataset/dataset.h5",
    label_name = "viticulture",
    label_format = "ordinal",
    mode = "test"
)

scalar_dataloader = DataLoader(ordinal_dataset, batch_size=1, shuffle=False)

masks, inputs, labels = next(iter(scalar_dataloader))

print(masks.shape, inputs.shape, labels.shape)
print("Converting ordinal encoded labels to actual class names:")
print()
labels = torch.sum(labels, axis=1) 
print(f"Label value is {int(labels[0,13,13])}, which corresponds to: {label_names[int(labels[0,13,13])]}")

```

    torch.Size([1, 128, 128]) torch.Size([1, 11, 10, 128, 128]) torch.Size([1, 4, 128, 128])
    Converting ordinal encoded labels to actual class names:
    
    Label value is 4, which corresponds to: very high


## You reach the end. Now you've got a flexible dataloader that is suitable for 3 machine learning tasks. 
## More formats can of course be derived
