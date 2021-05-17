import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score  # R2 coefficient of determination (goodness of fit)
import torch.utils.data as Data
import os

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Initialization of parameters
torch.manual_seed(1)


# Define data normalization functions
def normalization_height(num):
    return (num - 4.5) // 0.5


# Load the data
freq = pd.read_excel("Mode1.xlsx")
dataset_x = StandardScaler().fit_transform(pd.DataFrame(freq[["Electric_Abs_3D", "Magnetic_Abs_3D", "Mode 1"]].values))
# dataset_y = pd.DataFrame(freq[["cH", "cR"]])
dataset_y = pd.DataFrame(freq[["cH"]])
print(len(set(list(dataset_y["cH"].values))))
dataset_y = [[i] for i in dataset_y["cH"]]

# Data split
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.1, random_state=0)

x_train = torch.FloatTensor(x_train).type(torch.FloatTensor)
y_train = torch.FloatTensor(y_train).type(torch.FloatTensor)
x_test = torch.FloatTensor(x_test).type(torch.FloatTensor)
y_test = torch.FloatTensor(y_test).type(torch.FloatTensor)

# Batch training
BATCH_SIZE = 500

torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the neural net(4-layer)
net = torch.nn.Sequential(
    torch.nn.Linear(3, 2048),
    torch.nn.ReLU(),
    torch.nn.Linear(2048, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)

# Train the net
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)     # optimization function
loss_func = torch.nn.MSELoss()                              # loss function

r2_score_tem = 0      # Initialization of R2 coefficient

for epoch in range(2000):
    for step, (batch_x, batch_y) in enumerate(loader):
        prediction = net(batch_x)
        # print(prediction.size())
        # print(y_train.size())
        loss = loss_func(prediction, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch:", epoch, "Step:", step, "Loss:", loss.data.numpy())

# calculate R2 score of Test-set
r2_score_num = r2_score(net(x_test).data.numpy(), y_test.data.numpy())
if r2_score_num > r2_score_tem:
    r2_score_tem = r2_score_num
    torch.save(net, 'net_regression.pkl')  # R2 score: 0.928
net.train()

# Test the net
print(net(x_test).data.numpy())
print(y_test.data.numpy())
print(loss_func(net(x_test), y_test).data.numpy())
print("R2 score: ", r2_score_tem)
