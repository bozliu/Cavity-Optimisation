# Cavity Optimisation With Incorporation of Machine Learning Techniques
The objective of this project is to use given three attriabutes,  resonant frequency, electric field and magnetic field to predict height and radius.

## Dependencies and Installation

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- Pandas
-Sklearn
    ```
## Dataset
- File: Model1.xlsx
- Objective: Using given "Electric_Abs_3D", "Magnetic_Abs_3D", "Mode 1" to predict "cR" and "cH".
    
## Algorithem Design 
### Overview Structure of Software Design
<img src="https://github.com/bozliu/Cavity-Optimisation/blob/main/pic/Overview%20of%20System%20Architecture.png" width="50%">

### (1) Decision Tree
```
python decision_tree.py
```
<img src="https://github.com/bozliu/Cavity-Optimisation/blob/main/pic/Decision%20Tree%20Algorithm%20Design.png" width="50%">

### (2) Random Forest 
```
python random_forest.py
```
<img src="https://github.com/bozliu/Cavity-Optimisation/blob/main/pic/Random%20Forest%20Algorithm%20Design.png" width="50%">

### (3) Neural Network Classification (3 layers)

```
python NN_classfication.py
```
### (4) Nerual Network Regression (3 layers) 

```
python NN_regression.py
```
<img src="https://github.com/bozliu/Cavity-Optimisation/blob/main/pic/Neural%20Network%20Algorithm%20Design.png" width="50%">

## Code of Different Regression Methods
```
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
 ```

