from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


# Data normalization
def normalization_radius(num):
    return (num - 3.5) // 0.5


def normalization_height(num):
    return (num - 4.5) // 0.5


if __name__ == '__main__':
    freq = pd.read_excel("Mode1.xlsx")
    dataset_x = pd.DataFrame(freq[["Electric_Abs_3D", "Magnetic_Abs_3D", "Mode 1"]])
    # dataset_y = pd.DataFrame(freq[["cH", "cR"]])
    dataset_y = pd.DataFrame(freq[["cR"]])
    dataset_y = [normalization_radius(i) for i in dataset_y["cR"]]

    x_train, x_test, y_train, y_test = train_test_split(dataset_x.values, dataset_y, test_size=0.2,
                                                        random_state=44)
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy", (y_test == y_pred).sum() / len(x_test))
