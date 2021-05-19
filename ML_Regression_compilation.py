from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor


# Data normalization
def normalization_radius(num):
    return (num - 3.5) // 0.5


def normalization_height(num):
    return (num - 4.5) // 0.5


def try_different_method(model):
    model.fit(x_train,y_train)
    result = model.predict(x_test)
    score = r2_score(y_test, result)
    print("R2 score:", score)


if __name__ == '__main__':
    freq = pd.read_excel("Mode1.xlsx")
    dataset_x = pd.DataFrame(freq[["Electric_Abs_3D", "Magnetic_Abs_3D", "Mode 1"]])
    # dataset_y = pd.DataFrame(freq[["cH", "cR"]])
    dataset_y = pd.DataFrame(freq[["cH"]])
    dataset_y = [[i] for i in dataset_y["cH"]]

    x_train, x_test, y_train, y_test = train_test_split(dataset_x.values, dataset_y, test_size=0.2,
                                                        random_state=44)
    # Linear Regression
    linear_reg = linear_model.LinearRegression()
    # Decision Tree Regression
    decision_tree_reg = tree.DecisionTreeRegressor()
    # KNN Regression
    knn_reg = neighbors.KNeighborsRegressor()
    # Random Forest Regression
    random_forest_reg = ensemble.RandomForestRegressor(n_estimators=20)
    # Ada boost Regression
    ada_boost_reg = ensemble.AdaBoostRegressor(n_estimators=50)
    # GBRT Regression
    GBRT_reg = ensemble.GradientBoostingRegressor(n_estimators=100)
    # Bagging Regression
    bagging_reg = BaggingRegressor()
    # Extra Regression
    extra_tree_reg = ExtraTreeRegressor()

    try_different_method(extra_tree_reg)
