# ⚙️ MKT3434_2025

**MKT3434 Course of Dept. Mechatronics Eng. at YTU instructed by Ertugrul Bayraktar**

---

## 🚀 Overview

This repository provides a GUI framework with enhanced Classical-ML methods. In this updated GUI, regression and classification sections has been developed. 

---


## 🏁 Getting Started

### ⚙️ Prerequisites:

Ensure you have the following installed:

*   Python 3.8+

### 📦 Required dependencies:

```bash
pip install numpy pandas matplotlib PyQt6 scikit-learn tensorflow torch torchvision torchaudio opencv-python opencv-contrib-python scipy fastai kornia
```
## Models and new features
Note: In order to explain the parameters practically, there are parenthetical usage forms next to the parameters that carry meanings about how they are used. These usage patterns are explained here.

- (checkbox): User ticks the box to use this feature.
- (integer): User enters a desired integer to use this feature.
- (text): User enters a desired text to use this feautre.
- (double): User can enter a decimal number for this feature.
- (dropdown): User can choose the desired features that defined in the dropdown tab.

### Regression Models
For the regression models, loss functions of MSE (Mean Square Error), MAE(Mean Absolute Error) and Huber Loss is provided.

#### 1. Linear Regression
Linear Regression is a simple regression model that predicts a continuous target variable based on a linear relationship with the input features.

**Parameters:**
- `fit_intercept` (checkbox): Whether to calculate the intercept for the model. If `False`, no intercept will be used in the calculation.
- `n_jobs` (integer): The number of jobs to use for computation. `-1` means using all processors.
- `positive` (checkbox): When set to `True`, forces the coefficients to be positive.
- `feature_names` (text): Comma-separated names of the input features.
- `target_name` (text): Name of the target variable.
- `test_size` (double): Proportion of the dataset to include in the test split.

---

#### 2. Logistic Regression
Logistic Regression is a classification model that predicts the probability of a target variable belonging to a particular class.

**Parameters:**
- `C` (double): Inverse of regularization strength. Smaller values specify stronger regularization.
- `max_iter` (integer): Maximum number of iterations for the solver to converge.
- `multi_class` (dropdown): Specifies the type of classification. Options are:
  - `ovr`: One-vs-Rest (default).
  - `multinomial`: Multinomial logistic regression.
- `penalty` (dropdown): Specifies the norm used in the penalization. Options are:
  - `l1`: L1 regularization.
  - `l2`: L2 regularization. It is the default regularization.
  - `elasticnet`: Elastic Net regularization.
  - `none`: No regularization.
- `solver` (dropdown): Algorithm to use in the optimization problem. Options include:
  - `lbfgs`, `newton-cg`, `liblinear`, `sag`, `saga`.
- `class_weight` (dropdown): Weights associated with classes. Options are:
  - `balanced`: Adjust weights inversely proportional to class frequencies.
  - `none`: No weighting.
- `random_state` (integer): Seed for random number generation.
- `verbose` (integer): Verbosity level for the solver.
- `tol` (double): Tolerance for stopping criteria.
- `warm_start` (checkbox): Reuse the solution of the previous call to fit as initialization.
- `feature_names` (text): Comma-separated names of the input features.
- `test_size` (double): Proportion of the dataset to include in the test split.
- `l1_ratio` (double): Elastic Net mixing parameter, with `0 <= l1_ratio <= 1`.

---

#### 3. Support Vector Regression (SVR)
Support Vector Regression is a regression model that uses Support Vector Machines (SVM) to predict a continuous target variable.

**Parameters:**
- `C` (double): Regularization parameter. The strength of the regularization is inversely proportional to `C`.
- `kernel` (dropdown): Specifies the kernel type to be used in the algorithm. Options are:
  - `linear`: Linear kernel.
  - `rbf`: Radial Basis Function kernel (default).
  - `poly`: Polynomial kernel.
- `epsilon` (double): Specifies the epsilon-tube within which no penalty is associated in the training loss function with predictions.
- `feature_names` (text): Comma-separated names of the input features.
- `target_name` (text): Name of the target variable.
- `test_size` (double): Proportion of the dataset to include in the test split.

---

### Classification Models
For the classification models, Cross-Entropy and Hinge Loss functions are provided.

#### 1. Naive Bayes
Naive Bayes is a probabilistic classification model based on Bayes' theorem, assuming independence between features.

**Parameters:**
- `var_smoothing` (double): Portion of the largest variance of all features added to variances for numerical stability.
- `priors_type` (dropdown): Type of prior distribution. Options are:
  - `uniform`: Uniform prior distribution.
  - `custom`: Custom prior probabilities.
- `custom_priors` (text): Custom prior probabilities as comma-separated values (used if `priors_type` is set to `custom`).
- `feature_names` (text): Comma-separated names of the input features.
- `target_name` (text): Name of the target variable.

---

#### 2. Support Vector Machine (SVM)
Support Vector Machine is a classification model that finds the hyperplane that best separates the classes in the feature space.

**Parameters:**
- `C` (double): Regularization parameter. The strength of the regularization is inversely proportional to `C`.
- `kernel` (dropdown): Specifies the kernel type to be used in the algorithm. Options are:
  - `linear`: Linear kernel.
  - `rbf`: Radial Basis Function kernel (default).
  - `poly`: Polynomial kernel.
  - `sigmoid`: Sigmoid kernel.
- `degree` (integer): Degree of the polynomial kernel function (used if `kernel` is `poly`).
- `gamma` (dropdown or double): Kernel coefficient for `rbf`, `poly`, and `sigmoid`. Options are:
  - `scale`: Default value.
  - `auto`: Automatic scaling.
- `coef0` (double): Independent term in kernel function (used for `poly` and `sigmoid`).
- `class_weight` (dropdown): Weights associated with classes. Options are:
  - `balanced`: Adjust weights inversely proportional to class frequencies.
  - `none`: No weighting.
- `tol` (double): Tolerance for stopping criteria.
- `max_iter` (integer): Maximum number of iterations for the solver to converge.

---

#### 3. Decision Tree
Decision Tree is a classification model that splits the data into subsets based on feature values, forming a tree structure to make predictions.

**Parameters:**
- `criterion` (dropdown): The function to measure the quality of a split. Options are:
  - `gini`: Gini impurity.
  - `entropy`: Information gain.
- `max_depth` (integer): The maximum depth of the tree. If `None`, nodes are expanded until all leaves are pure or contain fewer than `min_samples_split` samples.
- `min_samples_split` (integer): The minimum number of samples required to split an internal node.
- `min_samples_leaf` (integer): The minimum number of samples required to be at a leaf node.
- `max_features` (dropdown): The number of features to consider when looking for the best split. Options are:
  - `sqrt`: Square root of the total number of features.
  - `log2`: Logarithm base 2 of the total number of features.
  - `none`: No restriction.
- `random_state` (integer): Seed for random number generation.
- `class_weight` (dropdown): Weights associated with classes. Options are:
  - `balanced`: Adjust weights inversely proportional to class frequencies.
  - `none`: No weighting.

---

#### 4. Random Forest
Random Forest is an ensemble classification model that builds multiple decision trees and combines their predictions for better accuracy and robustness.

**Parameters:**
- `criterion` (dropdown): The function to measure the quality of a split. Options are:
  - `gini`: Gini impurity.
  - `entropy`: Information gain.
- `n_estimators` (integer): The number of trees in the forest.
- `max_depth` (integer): The maximum depth of the tree. If `None`, nodes are expanded until all leaves are pure or contain fewer than `min_samples_split` samples.
- `min_samples_split` (integer): The minimum number of samples required to split an internal node.
- `min_samples_leaf` (integer): The minimum number of samples required to be at a leaf node.
- `max_features` (dropdown): The number of features to consider when looking for the best split. Options are:
  - `sqrt`: Square root of the total number of features.
  - `log2`: Logarithm base 2 of the total number of features.
  - `none`: No restriction.
- `bootstrap` (checkbox): Whether bootstrap samples are used when building trees.
- `random_state` (integer): Seed for random number generation.
- `class_weight` (dropdown): Weights associated with classes. Options are:
  - `balanced`: Adjust weights inversely proportional to class frequencies.
  - `none`: No weighting.
- `max_leaf_nodes` (integer): The maximum number of leaf nodes in the tree.
- `min_impurity_decrease` (double): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

---

#### 5. K-Nearest Neighbors (KNN)
K-Nearest Neighbors is a classification model that predicts the class of a data point based on the majority class of its nearest neighbors.

**Parameters:**
- `n_neighbors` (integer): The number of neighbors to use for classification.
- `metric` (dropdown): The distance metric to use. Options are:
  - `euclidean`: Euclidean distance.
  - `manhattan`: Manhattan distance.
  - `minkowski`: Minkowski distance.
  - `chebyshev`: Chebyshev distance.
  - `mahalanobis`: Mahalanobis distance.
  - `custom`: Custom distance function (not yet supported).
- `weights` (dropdown): Weight function used in prediction. Options are:
  - `uniform`: All neighbors are weighted equally.
  - `distance`: Closer neighbors are weighted more heavily.
- `algorithm` (dropdown): Algorithm used to compute the nearest neighbors. Options are:
  - `auto`: Automatically selects the best algorithm.
  - `ball_tree`: Ball Tree algorithm.
  - `kd_tree`: KD Tree algorithm.
  - `brute`: Brute-force search.
- `leaf_size` (integer): Leaf size passed to the Ball Tree or KD Tree algorithm.

---
