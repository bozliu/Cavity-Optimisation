import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import os

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Initialization of parameters
torch.manual_seed(1)  # reproducible


# Define data normalization functions
def normalization_height(num):
    return (num - 4.5) // 0.5


def normalization_radius(num):
    return (num - 3.5) // 0.5


# Load the data
freq = pd.read_excel("Mode1.xlsx")
dataset_x = StandardScaler().fit_transform(pd.DataFrame(freq[["Electric_Abs_3D", "Magnetic_Abs_3D", "Mode 1"]].values))
# dataset_y = pd.DataFrame(freq[["cH", "cR"]])
dataset_y = pd.DataFrame(freq[["cR"]])
print(len(set(list(dataset_y["cR"].values))))
dataset_y = [normalization_radius(i) for i in dataset_y["cR"]]

# enc = OneHotEncoder(sparse=False)
# enc.fit(dataset_y)
# dataset_y = enc.transform(dataset_y)
# print(dataset_y.shape)

x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.1, random_state=0)
x_train = torch.FloatTensor(x_train).type(torch.FloatTensor)
y_train = torch.FloatTensor(y_train).type(torch.LongTensor)
x_test = torch.FloatTensor(x_test).type(torch.FloatTensor)
y_test = torch.FloatTensor(y_test).type(torch.LongTensor)

# Define the neural net(cH has 93 classes and cR has 34 class) with 3 hidden layers
# function Relu process the data and propagate the data to next layer
net = torch.nn.Sequential(
    torch.nn.Linear(3, 1024),     # 3 inputs and 1024 hidden neurons
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),   # 256 0.66  512 0.4
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),    # 128 0.35
    torch.nn.ReLU(),
    torch.nn.Linear(512, 93)      # output layer
)

# Train the nets
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # optimization function
loss_func = torch.nn.CrossEntropyLoss()                # loss function

for t in range(10000):
    prediction = net(x_train)
    # print(prediction.size())
    # print(y_train.size())
    loss = loss_func(prediction, y_train)  # calculate the difference/loss between prediction and real values
    optimizer.zero_grad()                  # set the gradient to zero
    loss.backward()                        # back-propagation of the loss
    optimizer.step()                       # optimize the parameters by minimum loss
    print("Loss:", loss.data.numpy())
    prediction = torch.max(prediction, 1)[1]  # [0] return the largest value
    pred_y = prediction.data.numpy()
    target_y = y_train.data.numpy()
    accuracy = (pred_y == target_y).sum() / target_y.size
    print("Accuracy", accuracy)
    print()


print(torch.max(net(x_test), 1)[1].data.numpy())
print(y_test.data.numpy())
print("Final Accuracy:", ((y_test.data.numpy() == torch.max(net(x_test), 1)[1].data.numpy()).sum() /
                          y_test.data.numpy().size))
