import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    """
    This class allows to compute a regression architecture based on Linear and hyper-parameters
    such as the number of neurons, the number of hidden layers, the kernel size, the activation
    of hidde layers
    """

    def __init__(self, input_dim: int, output_dim: int, params: dict,
                 last_activation: bool = True):
        super(Regressor, self).__init__()
        hidden_dim: int = params['hidden_dim']
        n_hidden: int = params['n_hidden']
        activation: str = params['activation']
        self.stack_layers = self.get_layers(
            input_dim, hidden_dim, n_hidden, activation)
        self.fcl = nn.Linear(hidden_dim, output_dim)
        # prices are supposed to be >=0, hence relu
        self.last_activation = last_activation
        self.compute_set()

    @staticmethod
    def linear(input_dim, hidden_dim):
        return nn.Linear(input_dim, hidden_dim)

    @staticmethod
    def relu():
        return nn.ReLU()

    @staticmethod
    def sigmoid():
        return nn.Sigmoid()

    @staticmethod
    def tanh():
        return nn.Tanh()

    @staticmethod
    def set_linear(stack_layers: list, input_dim, hidden_dim, activation: str):
        stack_layers.append(Regressor.linear(input_dim, hidden_dim))
        if activation == 'relu':
            act = Regressor.relu()
        elif activation == 'sigmoid':
            act = Regressor.sigmoid()
        elif activation == 'tanh':
            act = Regressor.tanh()
        stack_layers.append(act)
        return stack_layers

    @staticmethod
    def get_layers(input_dim: int, hidden_dim: int, n_hidden: int, activation: str):
        stack_layers = []
        stack_layers = Regressor.set_linear(
            stack_layers, input_dim, hidden_dim, activation)
        for i in range(n_hidden):
            input_dim = hidden_dim
            stack_layers = Regressor.set_linear(
                stack_layers, input_dim, hidden_dim, activation)
        return stack_layers

    def compute_set(self):
        for i, layer in enumerate(self.stack_layers):
            name = f'Layer_{str(i + 1).zfill(3)}'
            setattr(self, name, layer)

    def forward(self, x):
        for i, layer in enumerate(self.stack_layers):
            x = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
        x = self.fcl(x)
        x = self.relu()(x) if self.last_activation else x
        return x


class CNN(nn.Module):
    """
    This class allows to compute a multi-classification architecture based on Conv2d and hyper-parameters
    such as the number of neurons, the number of hidden layers, the kernel size, the activation
    of hidde layers
    """

    def __init__(self, input_shape: tuple, output_dim: int, params: dict, last_activation: bool):
        super(CNN, self).__init__()
        params['input_dim'] = 1
        params['output_dim'] = output_dim
        self.stack_layers = self.get_layers(params)
        block_output_shape = self.compute_output_block_shape(
            params['kernel_size'], params['n_hidden'], input_shape)
        self.fcl = nn.Linear(
            block_output_shape[0] * block_output_shape[1] * params['hidden_dim'], params['output_dim'])
        self.last_activation = last_activation
        self.compute_set()

    @staticmethod
    def conv2d(params):
        return nn.Conv2d(in_channels=params['input_dim'], out_channels=params['hidden_dim'],
                         kernel_size=params['kernel_size'], padding=0, dilation=1, stride=1)

    @staticmethod
    def linear(input_dim, output_dim):
        return nn.Linear(in_features=input_dim, out_features=output_dim)

    @staticmethod
    def relu():
        return nn.ReLU()

    @staticmethod
    def sigmoid():
        return nn.Sigmoid()

    @staticmethod
    def tanh():
        return nn.Tanh()

    @staticmethod
    def flatten():
        return nn.Flatten()

    @staticmethod
    def softmax():
        return nn.Softmax()

    @staticmethod
    def set_conv2d(stack_layers: list, params):
        stack_layers.append(CNN.conv2d(params))
        if params['activation'] == 'relu':
            act = CNN.relu()
        elif params['activation'] == 'sigmoid':
            act = CNN.sigmoid()
        elif params['activation'] == 'tanh':
            act = CNN.tanh()
        stack_layers.append(act)
        return stack_layers

    @staticmethod
    def get_layers(params):
        stack_layers = []
        stack_layers = CNN.set_conv2d(stack_layers, params)
        for _ in range(params['n_hidden']):
            params['input_dim'] = params['hidden_dim']
            stack_layers = CNN.set_conv2d(stack_layers, params)
        return stack_layers

    @staticmethod
    def compute_output_block_shape(kernel_size, n_layers, input_shape, padding=0, dilatation=1, stride=1):
        current_shape = [input_shape[0], input_shape[1]]
        for _ in range(n_layers + 1):
            # print(current_shape)
            current_shape[0] = (
                (current_shape[0] + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride) + 1
            current_shape[1] = (
                (current_shape[1] + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride) + 1
        current_shape = [int(item) for item in current_shape]
        return current_shape

    def compute_set(self):
        for i, layer in enumerate(self.stack_layers):
            name = f'Layer_{str(i + 1).zfill(3)}'
            setattr(self, name, layer)

    def forward(self, x):
        for i, _ in enumerate(self.stack_layers):
            x = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
            # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fcl(x)
        x = F.softmax(x, dim=1) if self.last_activation else x
        return x
