House Prices: Advanced Regression Techniques

Project Overview

This project solves the Kaggle competition **House Prices: Advanced Regression Techniques**

The goal is to predict the sale price of residential properties using information about the house, such as its quality, area, year of construction, garage, basement, neighborhood, and other characteristics

The project includes:

* data cleaning and missing-value handling;
* exploratory data analysis (EDA);
* feature analysis and feature engineering;
* comparison of several regression models;
* cross-validation;
* hyperparameter tuning;
* error analysis;
* generation and validation of a Kaggle submission file

The main target variable is `SalePrice`

\---

Dataset

The project uses the Kaggle **House Prices: Advanced Regression Techniques** dataset

The notebook reads:

* `train.csv` — training data with the target variable `SalePrice`;
* `test.csv` — test data used to generate predictions

The current notebook is configured for the Kaggle environment and uses the following input path:

```text
/kaggle/input/competitions/house-prices-advanced-regression-techniques/
```

\---

Workflow

The project follows this general pipeline:

```text
Data Loading
     ↓
Missing Value Handling
     ↓
Exploratory Data Analysis
     ↓
Feature Analysis
     ↓
Feature Engineering
     ↓
Preprocessing
     ↓
Cross-Validation
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Error Analysis
     ↓
Final Model
     ↓
Kaggle Submission
```

\---

Data Preprocessing

Missing values are handled according to the meaning of the features

Categorical features

For features where a missing value represents the absence of an element, missing values are replaced with `"None"`

Examples:

* `Alley`
* `MasVnrType`
* `BsmtQual`
* `BsmtCond`
* `BsmtExposure`
* `BsmtFinType1`
* `BsmtFinType2`
* `FireplaceQu`
* `GarageType`
* `GarageFinish`
* `GarageQual`
* `GarageCond`
* `PoolQC`
* `Fence`
* `MiscFeature`

Numerical features

Some missing numerical values represent the absence of the corresponding feature and are replaced with `0`

Examples:

* `MasVnrArea`
* `GarageYrBlt`
* `BsmtFullBath`
* `BsmtHalfBath`
* `BsmtFinSF1`
* `BsmtFinSF2`
* `BsmtUnfSF`
* `TotalBsmtSF`
* `GarageArea`
* `GarageCars`

`LotFrontage` is imputed using the median value for the corresponding `Neighborhood`, with a global training median as a fallback

Other categorical variables in the test set are filled using the most frequent value from the training data

The modeling pipeline additionally uses:

* median imputation for numerical variables;
* most-frequent imputation for categorical variables;
* one-hot encoding for categorical variables

\---

Exploratory Data Analysis

The target variable `SalePrice` is right-skewed, with high-value outliers

The notebook investigates relationships between `SalePrice` and several important features, including:

* `OverallQual`
* `GrLivArea`
* `GarageCars`
* `GarageArea`
* `TotRmsAbvGrd`
* `BsmtFullBath`

Correlation analysis and scatter plots are used to understand relationships between variables and identify highly correlated features

Two notable pairs of correlated features are:

* `GarageCars` and `GarageArea`
* `GrLivArea` and `TotRmsAbvGrd`

These relationships are tested experimentally during model development

\---

Target Transformation

Because `SalePrice` is strongly right-skewed, the target is transformed using:

```python
y\_log = np.log1p(y)
```

Model evaluation is performed on the log-transformed target using **Root Mean Squared Error (RMSE)**

Predictions are converted back to the original price scale with:

```python
np.expm1(predictions)
```

\---

Feature Engineering

Several additional features were tested

`TotalSF`

Total square footage:

```text
TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

`GarageInteraction`

Interaction between garage capacity and garage area:

```text
GarageCars × GarageArea
```

`HouseAge`

Age of the house at the time of sale:

```text
YrSold - YearBuilt
```

`YearsSinceRemod`

Years since the most recent remodeling:

```text
YrSold - YearRemodAdd
```

`QualCondInteraction`

Interaction between overall quality and overall condition:

```text
OverallQual × OverallCond
```

The experiments showed that some engineered features added little or no useful improvement. `HouseAge` produced the most noticeable improvement among the tested feature-engineering experiments, but the improvement was small

\---

Models

Several regression models were evaluated during the project:

Linear Regression

Used as the main baseline model

Ridge Regression

Tested with `RidgeCV` and several values of `alpha`

Decision Tree Regressor

Used as a tree-based baseline

Random Forest Regressor

A Random Forest model was evaluated and then tuned with `RandomizedSearchCV`

The search included parameters such as:

* `n\_estimators`
* `max\_depth`
* `min\_samples\_split`
* `min\_samples\_leaf`
* `max\_features`

Gradient Boosting Regressor

Gradient Boosting was also evaluated and tuned with `RandomizedSearchCV`

The search included:

* `n\_estimators`
* `learning\_rate`
* `max\_depth`
* `min\_samples\_split`
* `min\_samples\_leaf`
* `subsample`
* `max\_features`
* `loss`

The tuned Gradient Boosting model is used in the notebook to generate the final Kaggle predictions

\---

Model Evaluation

Models are evaluated using **5-fold cross-validation**:

```python
KFold(
    n\_splits=5,
    shuffle=True,
    random\_state=42
)
```

The main evaluation metric is:

```text
RMSE
```

calculated on `log1p(SalePrice)`

The notebook also compares out-of-fold predictions and prediction errors to understand where the models perform well and where they struggle

Exact CV values are intentionally not hard-coded in this README because they depend on the current preprocessing, feature set, and hyperparameter search results printed when the notebook is executed

\---

Error Analysis

Out-of-fold predictions are used to investigate model errors

The analysis looks at:

* actual vs predicted `log(SalePrice)`;
* residual distribution;
* absolute error;
* error by `OverallQual`;
* error by `GrLivArea`;
* error by `Neighborhood`;
* error across different price segments

The analysis indicates that large errors are concentrated around difficult or extreme price cases

For the tree-based models, the notebook also examines systematic residual patterns and possible overprediction of some groups of houses

\---

Final Prediction and Submission

Before generating the submission file, the notebook checks that train and test features are aligned

The final prediction workflow is:

```text
Train final model
      ↓
Predict log(SalePrice)
      ↓
Convert predictions with expm1()
      ↓
Create submission DataFrame
      ↓
Validate columns and predictions
      ↓
Save submission.csv
```

The submission file contains:

```text
Id,SalePrice
```

The notebook validates that:

* the `Id` column matches `test.csv`;
* predictions do not contain `NaN`;
* predictions are finite;
* predictions are not negative

The final file is saved as:

```text
submission.csv
```

\---

Project Structure

A simple repository structure can be:

```text
house-prices-regression/
│
├── README.md
├── notebook.ipynb
└── submission.csv
```

The exact notebook filename can be changed to match the file stored in the repository

\---

Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Kaggle

\---

Main Python Tools

The project uses:

* `pandas` for data manipulation;
* `numpy` for numerical operations and target transformation;
* `matplotlib` and `seaborn` for visualization;
* `scikit-learn` for preprocessing, cross-validation, regression models, and hyperparameter tuning

\---

Running the Project

Kaggle

The notebook is already configured for the Kaggle dataset path

Open the notebook in Kaggle and run the cells sequentially

Local environment

To run the project locally:

1. Download the Kaggle dataset
2. Place `train.csv` and `test.csv` in a local data directory
3. Replace the Kaggle-specific dataset path in the notebook
4. Install the required Python packages
5. Run the notebook from start to finish

Example:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

\---

Notes

One preprocessing step for `LotFrontage` is fitted using the full training set before cross-validation. For a strictly leakage-free validation pipeline, all preprocessing steps should be fitted separately inside each cross-validation fold

This project is primarily focused on the complete machine learning workflow: preprocessing, analysis, experimentation, validation, error analysis, and Kaggle submission generation

