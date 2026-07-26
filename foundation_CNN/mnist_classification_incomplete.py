"""

"""

import torch

from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, Flatten
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

"""
To download the data using torchvision - mnist is nothing but downloading the raw data set to put into root directory.
But this data won'be usable because you need to transform the data into tensor() - which is being done using
transforms.ToTensor() which always needed to do. because machine/training framework understands tensor in gpu/cpu.

original size of MNIST data image shape: (28, 28)
"""

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # which converts the images to tesnor
        transforms.Normalize(mean=0.13, std=0.1),  # normising the data
        transforms.Resize(size=(32, 32))  # resizing the image into (32, 32) [original image is in formation of ]

    ]
)

"""
if you download the data it is nothing but just the raw image but to use this for training and testing so we need to 
transform to transforms.ToTensor()

torchvision.dataset contains:
mnist data: 
EMNIST, 
FashionMNIST, 
KMNIST, 
MNIST, 
QMNIST
"""

train_mnist = mnist.MNIST(
    root="./data",
    download=True,
    train=True,
    transform=transform
)

test_mnist = mnist.MNIST(
    root="./data",
    download=True,
    train=False,
    transform=transform
)

print(len(train_mnist))  # this is downloaded
print(len(test_mnist))  # this is also downloaded

# data processing for training
train_dataloader = DataLoader(
    train_mnist,  # training data downloaed and transformed into tensor.
    batch_size=64,  # number of images to be processed in one time __getitem__() function of dataset class.
    shuffle=True,
    # It is for shuffling the data for training. It is important to shuffle the data for training to avoid overfitting.
    num_workers=0,
    # how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    pin_memory=torch.cuda.is_available(),
    # this is for pinning the memory to the GPU. It is important to pin the memory to the GPU to avoid data transfer bottleneck.
    persistent_workers=False,
    # this is for keeping the workers alive after the dataset has been consumed. It is important to keep the workers alive to avoid data loading bottleneck.
    drop_last=False,
    # this is for dropping the last batch if it is smaller than the batch size. It is important to drop the last batch to avoid data loading bottleneck.
    # prefetch_factor=1
    # this is for multiprocessing. It specifies the number of samples loaded in advance by each worker.
)

test_dataloader = DataLoader(
    test_mnist,
    batch_size=64,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    drop_last=False,
    # prefetch_factor=1
)

# now need to train the model so, need to define the model

"""
first of all, the torch.nn.sequential is for prototypes - It is just to show that this can be done using 
the fixed hard-coded input size. It can not be change. because there is no forward() function defined here, 
in the case of torch.nn.Sequential() just create a container module that chains multiple neural network layers together
in sequence order, and when data passes through a Sequential block, pytorch automatically feeds the output of the first 
layer to next layer. because again, forward() defined defined internally for this kind of implementation.

but torch.nn.Sequential() does have limitations: can't handle branching, multiple inputs, and skip connections
1. It can not be used for varying size of input. It should be hardcoded. because its underlying source code is hardcoded 
to execute layers using a strict Python for loop that accepts exactly one input and passes it to one output; in forward() 

# Conceptual look at PyTorch's internal source code for Sequential
def forward(self, input):
    for module in self._modules.values():
        input = module(input)  # Overwrites the variable every time
    return input

 
1. It Only Accepts One Argument: The loop variable input can only hold one tensor at a time. If you try to pass two 
separate tensors into a Sequential block (e.g., model(image, text)), the code will immediately crash because the loop
 signature only expects a single variable.
 
2.  It Instantly Forgets Past States (No Skip Connections)Because the line(81) input = module(input) overwrites the input
 variable at every single step, the network completely forgets what the data looked like two layers ago.
 To create a skip connection (like a ResNet), you need to remember an earlier state to add it back later:

3. It Follows a Single Pipeline (No Branching): The data must travel down a single, un-branching track. 
If your network needs to split the data (for example, sending an image down one path to detect bounding boxes and 
another path to detect colors), a simple linear for loop cannot split or manage those parallel tracks.

Three way of implemnenting architecture in Pytorch:
1. Using torch.nn.Sequential (Simplest)You chain layers together in a strict linear list. Data flows automatically from 
top to bottom.Best For: Quick prototypes, standard Feed-Forward networks, or basic CNNs.Limitations: Cannot handle models
 with multiple inputs/outputs, branching, or skip connections (like ResNets).pythonimport torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)


2. Subclassing torch.nn.Module (Most Flexible & Industry Standard)You define your layers inside the __init__ constructor
 and manually script how data flows through them inside the forward method.Best For: Complex architectures, ResNets, 
 Transformers, GANs, or any model requiring custom debugging (e.g., printing tensor shapes).
 
 Limitations: Requires writing more boilerplate code
 import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Manually direct the flow of data
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

model = MyNetwork()

 
3. 3. Subclassing torch.nn.Module with Internal Sequential Blocks (Hybrid Approach)You build a standard nn.Module class,
 but instead of tracking every tiny layer individually, you group repetitive blocks using nn.Sequential.
 Best For: Keeping large architectures organized, clean, and highly readable.
 
 Limitations: None; it provides the best of both worlds.

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Group feature extraction layers together
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Group classification layers together
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 13 * 13, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNN()


"""

# model = torch.nn.Sequential(
#     Conv2d(
#         in_channels=1,
#         out_channels=64,
#         kernel_size=(3, 3)),
#     BatchNorm2d(64),
#     ReLU(),
#     MaxPool2d(3, 3),
#     Conv2d(64, 128, (3,3)),
#     BatchNorm2d(128),
#     ReLU(),
#     MaxPool2d(3, 3),
#     Flatten(),
#     Linear(512, 10),
#     Softmax(dim=1)
# )

"""

mnist_model_sequential architecture trace for input shape (1, 32, 32)  [channels x height x width]
Table: Layer Type | Hyperparameters | Calculation | Learnable Parameters | Output Dimension

--------------------------------------------------------------------------------------------------------------------------------------------------------
#   | Layer Type   | Hyperparameters                                    | Calculation                                          | Learnable Parameters       | Output Dimension
--------------------------------------------------------------------------------------------------------------------------------------------------------
0   | Input        | -                                                  | -                                                    | 0                          | (1, 32, 32)
1   | Conv2d       | in=1, out=64, kernel=3x3, stride=1, padding=same   | O = floor(((32-3)+(2*1))/1) + 1 = 32                 | weights: 1*3*3*64=576      | (64, 32, 32)
    |              |                                                    |                                                      | bias: 64  -> total 640     |
2   | BatchNorm2d  | num_features=64, affine=True                       | normalizes per-channel, no spatial change            | gamma:64 + beta:64 = 128   | (64, 32, 32)
3   | ReLU         | -                                                  | elementwise, no shape change                         | 0                          | (64, 32, 32)
4   | MaxPool2d    | kernel=2x2, stride=2                               | O = floor(((32-2)+0)/2) + 1 = 16                     | 0                          | (64, 16, 16)
5   | Conv2d       | in=64, out=128, kernel=3x3, stride=1, padding=same | O = floor(((16-3)+(2*1))/1) + 1 = 16                 | weights: 64*3*3*128=73728  | (128, 16, 16)
    |              |                                                    |                                                      | bias: 128 -> total 73856   |
6   | BatchNorm2d  | num_features=128, affine=True                      | normalizes per-channel, no spatial change            | gamma:128 + beta:128 = 256 | (128, 16, 16)
7   | ReLU         | -                                                  | elementwise, no shape change                         | 0                          | (128, 16, 16)
8   | MaxPool2d    | kernel=2x2, stride=2                               | O = floor(((16-2)+0)/2) + 1 = 8                      | 0                          | (128, 8, 8)
9   | Flatten      | -                                                  | 128 * 8 * 8 = 8192                                   | 0                          | (8192,)
10  | Linear       | in_features=8192, out_features=1024                | weight matrix 8192 x 1024                            | weights: 8192*1024=8388608 | (1024,)
    |              |                                                    |                                                      | bias: 1024 -> total 8389632|
11  | ReLU         | -                                                  | elementwise, no shape change                         | 0                          | (1024,)
12  | Linear       | in_features=1024, out_features=256                 | weight matrix 1024 x 256                             | weights: 1024*256=262144   | (256,)
    |              |                                                    |                                                      | bias: 256 -> total 262400  |
13  | ReLU         | -                                                  | elementwise, no shape change                         | 0                          | (256,)
14  | Linear       | in_features=256, out_features=10                   | weight matrix 256 x 10                               | weights: 256*10=2560       | (10,)
    |              |                                                    |                                                      | bias: 10  -> total 2570    |
--------------------------------------------------------------------------------------------------------------------------------------------------------

Total learnable parameters = 640 + 128 + 73856 + 256 + 8389632 + 262400 + 2570 = 8,729,482

Observation: The two Linear layers (8192->1024 and 1024->256) dominate the parameter count (~99%),
while the convolutional feature-extraction layers are comparatively lightweight - a common pattern
when flattening large spatial feature maps directly into fully connected layers.

"""

mnist_model_sequential = torch.nn.Sequential(
    Conv2d(
        # it is a learnable parameter that takes in input and output channels, kernel size, stride, padding, etc. It is a 2D convolutional layer that applies a convolution operation to the input image.
        in_channels=1,
        out_channels=64,
        kernel_size=(3, 3),
        stride=1,
        padding="same"
        # padding="same" means that the output size will be the same as the input size. It is used to keep the spatial dimensions of the output the same as the input.
    ),

    BatchNorm2d(
        num_features=64,
        affine=True
        # this shows the learning parameter is true or false. If it is true, then the layer will learn the parameters. If it is false, then the layer will not learn the parameters.
    ),

    ReLU(),
    # it is a non-linear activation function that applies the rectified linear unit function to the input. It is used to introduce non-linearity to the model.

    MaxPool2d(
        kernel_size=2,
        # 2x2 pooling window : to find the maximum in that window and downsample the input image. It is used to reduce the spatial dimensions of the input image.
        stride=2
        # it is the step size of the pooling window. It is used to control the amount of downsampling. It is used to reduce the spatial dimensions of the input image.
    ),

    Conv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=(3, 3),
        stride=1,
        padding="same"
    ),
    BatchNorm2d(
        num_features=128,
        affine=True
    ),
    ReLU(),
    MaxPool2d(
        kernel_size=2,
        stride=2
    ),

    Flatten(),  # it is a layer that flattens the input image into a 1D vector.
    #          # It is used to prepare the input for the fully connected layer.
    # so basically here the after flatten() : the input size will be 128 * 8 * 8
    # so I am not directly going for 128 * 8 * 8
    # as torch.nn.Sequential is mostly for hardcoded value; It is important to keep track of the
    # output shape of the previous layer.

    Linear(
        in_features=128 * 8 * 8,
        out_features=1024  # I am just slowing downsampling to num_classes for the classification task - other wise It
        # it will be too much of training requires to understand
    ),

    ReLU(),

    Linear(
        in_features=1024,
        out_features=256
    ),

    ReLU(),
    
    Linear(
        in_features=256,
        out_features=10
    ),
    # now we have 10 classes of calculated value now we can use softmax() function for getting the classification output.

    # Softmax(dim=1) # not required If we are using cross entropy loss

)


class non_sequential_model(nn.Module):
    def __init__(self):
        super(non_sequential_model, self).__init__()

        self.conv2d_1 = Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
        )

        self.batch_norm_1 = BatchNorm2d(
            num_features=64,
            affine=True
        )
        self.relu_1 = ReLU()

        self.maxpool2d_1 = MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        self.conv2d_2 = Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding="same",
            stride=1
        )

        self.batch_norm_2 = BatchNorm2d(
            num_features=128,
            affine=True
        )

        self.relu_2 = ReLU()

        self.maxpool2d_2 = MaxPool2d(
            kernel_size=(2, 2),
            stride=2
        )

        self.flatten = Flatten()

        self.FC_Linear_1 = Linear(
            in_features=128 * 8 * 8,
            out_features=1024
            # I am just slowing downsampling to num_classes for the classification task - other wise It
            # it will be too much of training requires to understand
        )

        self.relu_3 = ReLU()

        self.FC_Linear_2 = Linear(
            in_features=1024,
            out_features=256
        )

        self.relu_4 = ReLU()

        self.FC_Linear_3 = Linear(
            in_features=256,
            out_features=10
        )

        self.soft_max = Softmax(dim=1) # not required bcz we are using cross entropy but can be added bcz I am not
                                        # using in forwad()

    def forward(self, input):
        output_from_conv2d1 = self.conv2d_1(input)
        output_from_batch_norm_1 = self.batch_norm_1(output_from_conv2d1)
        output_from_relu_1 = self.relu_1(output_from_batch_norm_1)
        output_from_maxpool_1 = self.maxpool2d_1(output_from_relu_1)
        output_from_conv2d2 = self.conv2d_2(output_from_maxpool_1)
        output_from_batch_norm_2 = self.batch_norm_2(output_from_conv2d2)
        output_from_relu_2 = self.relu_2(output_from_batch_norm_2)
        output_from_maxpool_2 = self.maxpool2d_2(output_from_relu_2)
        output_from_flatten = self.flatten(output_from_maxpool_2)
        output_from_Linear_1 = self.FC_Linear_1(output_from_flatten)
        output_from_relu_3 = self.relu_3(output_from_Linear_1)
        output_from_Linear_2 = self.FC_Linear_2(output_from_relu_3)
        output_from_relu_4 = self.relu_4(output_from_Linear_2)
        output_from_Linear_3 = self.FC_Linear_3(output_from_relu_4)
        return output_from_Linear_3

        # output_from_softmax_logits = self.soft_max(output_from_Linear_3) # this is not required because we are using crossentropy as loss
        # return output_from_softmax_logits
        # now we have 10 classes of calculated value now we can use softmax() function for getting the classification output.


class hybrid_sequential_nn_mnist(nn.Module):
    def __init__(self):
        super(hybrid_sequential_nn_mnist, self).__init__()
        self.feature_extractor_conv_layer_1 = nn.Sequential(
            Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(3, 3),
                padding="same",
                stride=1
            ),

            BatchNorm2d(
                num_features=64,
                affine=True
            ),

            ReLU(),

            MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            )

        )

        self.feature_extractor_conv_layer_2 = nn.Sequential(
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding="same",
                stride=1
            ),

            BatchNorm2d(
                num_features=128,
                affine=True
            ),

            ReLU(),

            MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            )

        )

        self.final_fully_connected_layers = nn.Sequential(
            Flatten(),
            Linear(
                in_features=128 * 8 * 8,
                out_features=1024
            ),

            ReLU(),

            Linear(
                in_features=1024,
                out_features=512
            ),

            ReLU(),

            Linear(
                in_features=512,
                out_features=10
            )

        )

        # self.final_logits = Softmax(dim=1) # It is not important If we are using crossentropy as loss.

    def forward(self, input):
        output_from_first_feature_layer = self.feature_extractor_conv_layer_1(input)
        output_from_second_feature_layer = self.feature_extractor_conv_layer_2(output_from_first_feature_layer)
        output_from_fully_connected_layers = self.final_fully_connected_layers(output_from_second_feature_layer)
        return output_from_fully_connected_layers
        # output_from_softmax_for_logits = self.final_logits(output_from_fully_connected_layers) # not using because we
                                                                                            # are using crossentropy()
        # return output_from_softmax_for_logits


mnist_model_sequential = mnist_model_sequential
mnist_model_nn_module = non_sequential_model()
mnist_model_sequential_with_nn_module = hybrid_sequential_nn_mnist()

# Hyper parameters
lr = 0.001
epochs = 10
loss_function = CrossEntropyLoss()


def train_one_epoch(model, optimiser, dataloader, device):
    model.train()
    total_loss = 0.0

    for image, labels in dataloader:
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        prediction = model(image)
        loss_value = loss_function(prediction, labels)
        total_loss += loss_value.item()
        loss_value.backward()
        optimiser.step()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for image, labels in dataloader:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            prediction = model(image)
            loss_value = loss_function(prediction, labels)
            total_loss += loss_value.item()

            predicted_labels = prediction.argmax(dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def run_training_for_model(model_name, model, device):
    print(f"\n===== Training {model_name} =====")
    model = model.to(device)
    optimiser = Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        train_loss = train_one_epoch(model, optimiser, train_dataloader, device)
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, device)
        print(
            f"Epoch {i + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_registry = [
        ("mnist_model_sequential", mnist_model_sequential),
        ("mnist_model_nn_module", mnist_model_nn_module),
        ("mnist_model_sequential_with_nn_module", mnist_model_sequential_with_nn_module),
    ]

    for model_name, selected_model in model_registry:
        run_training_for_model(model_name, selected_model, device)
