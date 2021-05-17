import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Define data normalization functions
def normalization_height(num):
    return (num - 4.5) // 0.5


def normalization_radius(num):
    return (num - 3.5) // 0.5


if __name__ == '__main__':
    freq = pd.read_excel("Mode1.xlsx")    # Data loading
    dataset_x = pd.DataFrame(freq[["Electric_Abs_3D", "Magnetic_Abs_3D", "Mode 1"]])
    # dataset_y = pd.DataFrame(freq[["cH", "cR"]])
    dataset_y = pd.DataFrame(freq[["cR"]])
    dataset_y = [normalization_radius(i) for i in dataset_y["cR"]]  # data normalization
    # data split
    x_train, x_test, y_train, y_test = train_test_split(dataset_x.values, dataset_y, test_size=0.1, random_state=44)


    rfc = RandomForestClassifier()
    rfc = rfc.fit(x_train, y_train)
    result = rfc.score(x_test, y_test)
    print("Accuracy", result)
