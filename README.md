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
<img src="https://github.com/bozliu/E2E-Keyword-Spotting/blob/main/images/Standard%20E2E%20Architecture.png " width="100%">

### (1) Decision Tree
```
python decision_tree.py
```
<img src="https://github.com/bozliu/E2E-Keyword-Spotting/blob/main/images/Multi-Head%20Attention%20Architecture.png" width="100%">

### (2) Random Forest 
```
python random_forest.py
```

### (3) Neural Network Classification (3 layers)

```
python NN_classfication.py
```
### (4) Nerual Network Regression (3 layers) 

```
python NN_regression.py
```


