# PyTorch Abstraction Layer

PyTorch Abstraction Layer is an open-source project that provides a high-level API for PyTorch, a popular deep learning framework. The goal of this project is to simplify and abstract the low-level PyTorch API and make it easier for users to quickly build and train deep learning models.

## Local install for devs

To use this repository as a normal python package and still be able to dev on it, the suitable way of doing this would be by uising a conda env and then clone this repository in a folder and then use this command once you are in the repository folder: `pip install -e .`

## Install from link

Simply run this command: `pip install https://github.com/asarfa/PyTorchModels` This is currently not working

## Firsts steps into the easy life ...

```python
from asfarapi import Trainer
transforms = [
    RandomRotation([-15, 15]),
    RandomTranslate([-25, 25], [-25, 25])
]
train_data = DataGenerator("./data/train")
test_data = DataGenerator("./data/validation")
train_loader = DataLoader(train_data, 4, shuffle=True)
test_loader = DataLoader(test_data, 4)
model = UNet(12, 1)
trainer = Trainer(
    model=model,
    epochs=400,
    lr=3e-5,
    criterion=torch.nn.L1Loss(),
    transforms=transforms,
    opt='Adam',
    save_path="./models/parallel_mask_train",
    verbose=True
)
print("Fitting the model ...")
trainer.fit(train_loader, test_loader)
```

## Key Features

- A simple and intuitive generic classes for building complex deep learning models
- Convenient utility classes for common deep learning tasks such as data loading, data preprocessing, and model training
- Abstraction layer for PyTorch modules and optimizers to make it easier to work with complex models

## Installation

You can use this classes by cloning this repo on your computer and importing the classes you want

## Contributions

PyTorch Abstraction Layer is an open-source project and we welcome contributions from the community. If you would like to contribute, please read the `CONTRIBUTING.md` file for more information.

## License

PyTorch Abstraction Layer is released under the MIT License. Please see the `LICENSE` file for more information.
