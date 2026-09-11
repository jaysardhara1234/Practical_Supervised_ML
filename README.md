# 🏠 House Price Prediction --- Ames Housing

> 📌 **End-to-end regression project** for predicting house sale prices
> using the Ames Housing Dataset.

This project covers the complete machine-learning workflow: **problem
framing → EDA → preprocessing → feature engineering → regression
modeling → cross-validation → hyperparameter tuning → residual analysis
→ model saving**.

------------------------------------------------------------------------

## 📌 Table of Contents

-   [🎯 Project Objective](#-project-objective)
-   [📊 Dataset](#-dataset)
-   [🧠 Problem Framing](#-problem-framing)
-   [🔎 Exploratory Data Analysis](#-exploratory-data-analysis)
-   [🛠️ Data Preprocessing](#️-data-preprocessing)
-   [⚙️ Feature Engineering](#️-feature-engineering)
-   [🔤 Encoding and Scaling](#-encoding-and-scaling)
-   [🤖 Models](#-models)
-   [📈 Model Performance](#-model-performance)
-   [🔍 Cross-Validation](#-cross-validation)
-   [🎛️ Hyperparameter Tuning](#️-hyperparameter-tuning)
-   [🏆 Final Model](#-final-model)
-   [🧪 Residual Analysis](#-residual-analysis)
-   [💡 Feature Interpretation](#-feature-interpretation)
-   [📦 Saved Model](#-saved-model)
-   [📁 Project Files](#-project-files)
-   [🚀 How to Run](#-how-to-run)
-   [🧰 Technologies Used](#-technologies-used)
-   [📌 Key Learnings](#-key-learnings)

------------------------------------------------------------------------

## 🎯 Project Objective

The objective is to build a regression model that predicts a property's
**SalePrice** from its structural, quality, location, garage, basement,
and other housing characteristics.

### Target Variable 🎯

**`SalePrice`** --- the final sale price of the house.

Because the original target is right-skewed, the project uses:

``` python
y_log = np.log1p(y)
```

Predictions are converted back to the original price scale using:

``` python
y_pred = np.expm1(y_pred_log)
```

------------------------------------------------------------------------

## 📊 Dataset

The supplied Ames Housing Dataset contains:

-   🏘️ **1,460 observations**
-   📋 **81 columns**
-   🔢 **38 numerical columns**
-   🔤 **43 categorical columns**
-   🎯 **`SalePrice`** as the target variable
-   ⚠️ **19 columns containing missing values**

The dataset contains information about:

-   Lot and property characteristics
-   Overall quality and condition
-   Year built and remodeling
-   Basement features
-   Living area
-   Bathrooms and bedrooms
-   Kitchen and fireplace characteristics
-   Garage characteristics
-   Outdoor/pool features
-   Sale information

------------------------------------------------------------------------

## 🧠 Problem Framing

### Regression vs Classification

This is a **supervised regression problem** because the target is a
continuous numerical value.

-   📈 **Regression:** predicts a continuous value such as house price.
-   🏷️ **Classification:** predicts a categorical class such as
    spam/not-spam.

In simple terms:

> **Regression answers: "How much?"**\
> **Classification answers: "Which category?"**

### Business Context 💼

Accurate house-price prediction can help:

-   🏠 Estimate property values
-   💰 Support pricing decisions
-   📊 Compare properties objectively
-   🤝 Assist buyers, sellers, and real-estate businesses

------------------------------------------------------------------------

## 🔎 Exploratory Data Analysis

The notebook performs several stages of EDA.

### 1️⃣ Dataset Understanding

Basic inspection was performed using:

``` python
df.info()
df.describe()
```

This identified the dataset dimensions, data types, numerical
distributions, and missing values.

### 2️⃣ Missing-Value Analysis ⚠️

The following 19 columns contain missing values:

``` text
LotFrontage
Alley
MasVnrType
MasVnrArea
BsmtQual
BsmtCond
BsmtExposure
BsmtFinType1
BsmtFinType2
Electrical
FireplaceQu
GarageType
GarageYrBlt
GarageFinish
GarageQual
GarageCond
PoolQC
Fence
MiscFeature
```

### 3️⃣ Target Distribution 📉

`SalePrice` is right-skewed and does not closely follow a normal
distribution.

A `log1p` transformation was therefore used to make the target
distribution more symmetric and suitable for regression modeling.

### 4️⃣ Categorical Analysis 🔤

ANOVA was performed on categorical variables to investigate their
relationship with `SalePrice`.

The notebook identified influential categorical variables including:

-   `ExterQual`
-   `KitchenQual`
-   `BsmtQual`
-   `GarageFinish`

### 5️⃣ Correlation Analysis 🔗

Numerical features were examined through correlation with `SalePrice`,
including a heatmap of the top correlated numerical variables.

### 6️⃣ Outlier Analysis 🚨

Two observations with extremely large `GrLivArea` but unusually low
`SalePrice` were identified as clear outliers.

These observations were considered capable of distorting a regression
model because they do not follow the general positive relationship
between living area and sale price.

------------------------------------------------------------------------

## 🛠️ Data Preprocessing

The project uses an **80/20 train-test split**:

``` python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)
```

The target is transformed after splitting:

``` python
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
```

### 🗑️ Columns Removed

The following columns were dropped because of their very high proportion
of missing values:

``` python
['PoolQC', 'MiscFeature', 'Alley', 'Fence']
```

### 🧩 Missing-Value Strategies

Different missing-value strategies were used according to the meaning of
the variables.

#### 🏚️ Categorical features

Missing values in features such as garage, basement, fireplace, and
masonry-related variables were replaced with:

``` python
SimpleImputer(
    strategy='constant',
    fill_value='None'
)
```

This represents the absence of the corresponding feature.

#### 🔢 Numerical features

`GarageYrBlt` and `MasVnrArea` were filled with:

``` python
SimpleImputer(
    strategy='constant',
    fill_value=0
)
```

#### 📐 `LotFrontage`

Median imputation was used:

``` python
SimpleImputer(strategy='median')
```

#### ⚡ `Electrical`

Most-frequent imputation was used:

``` python
SimpleImputer(strategy='most_frequent')
```

------------------------------------------------------------------------

## ⚙️ Feature Engineering

Several meaningful features were created.

### 🏠 TotalSF

Total finished/basement-related square footage:

``` python
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

### 🎂 HouseAge

Age of the house at the time of sale:

``` python
HouseAge = YrSold - YearBuilt
```

### 🔨 RemodAge

Years since the most recent remodeling:

``` python
RemodAge = YrSold - YearRemodAdd
```

### 🚗 HasGarage

Binary indicator showing whether the property has a garage:

``` python
HasGarage = (GarageArea > 0).astype(int)
```

### 🏊 HasPool

Binary indicator showing whether the property has a pool:

``` python
HasPool = (PoolArea > 0).astype(int)
```

------------------------------------------------------------------------

## 🔤 Encoding and Scaling

### 📊 Ordinal Encoding

Quality-related variables were manually mapped because their categories
have a natural order:

``` text
None → 0
Po   → 1
Fa   → 2
TA   → 3
Gd   → 4
Ex   → 5
```

Applied to:

``` text
ExterQual
KitchenQual
BsmtQual
FireplaceQu
```

### 📉 Skewness Transformation

Numerical features with skewness greater than `0.75` were transformed
using:

``` python
np.log1p()
```

### ⚖️ Standardization

Continuous numerical features were standardized using:

``` python
StandardScaler()
```

### 🔤 One-Hot Encoding

Categorical variables were encoded using:

``` python
OneHotEncoder(handle_unknown='ignore')
```

The transformed data was combined using:

``` python
ColumnTransformer(...)
```

`Neighborhood` was retained even though it has 25 categories because the
notebook found it to have strong predictive value and considered its
cardinality manageable for One-Hot Encoding.

------------------------------------------------------------------------

## 🤖 Models

The project compares five regression approaches:

1.  📏 **Linear Regression**
2.  🧲 **Ridge Regression (L2)**
3.  ✂️ **Lasso Regression (L1)**
4.  🌲 **Random Forest Regressor**
5.  🚀 **XGBoost Regressor**

The models are trained on the **log-transformed SalePrice** and
evaluated after converting predictions back to the original price scale.

------------------------------------------------------------------------

## 📈 Model Performance

### 🏁 Test-Set Comparison

  Model                    RMSE ($) | MAE ($)              R² 
  ---------------------- -------------------- --------------- ------------
  📏 Linear Regression              20,767.52       14,587.42       0.9157
  🧲 Ridge (L2)                 **19,269.88**   **13,768.06**   **0.9274**
  ✂️ Lasso (L1)                     19,799.02       14,423.63       0.9234
  🌲 Random Forest                  24,059.37       17,008.84       0.8869
  🚀 XGBoost                        20,758.07       13,965.48       0.9158

### 🏆 Best Untuned Model

**Ridge Regression** achieved the strongest overall test performance:

-   🥇 Lowest RMSE: **\$19,269.88**
-   🥇 Lowest MAE: **\$13,768.06**
-   🥇 Highest R²: **0.9274**

This means the Ridge model explained approximately **92.74% of the
variance** in the test-set SalePrice.

------------------------------------------------------------------------

## 🔍 Cross-Validation

Five-fold shuffled K-Fold cross-validation was performed for Ridge
Regression:

``` python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=10
)
```

### Results

-   📊 Mean CV RMSE: **\$22,114.64**
-   📏 Standard deviation: **\$624.15**

Individual fold RMSE values:

``` text
22,696.77
21,145.84
22,724.29
21,633.14
22,373.15
```

The CV RMSE is higher than the test RMSE of `$19,269.88`, suggesting
that the particular test split was relatively easier than the average
validation split.

------------------------------------------------------------------------

## 🎛️ Hyperparameter Tuning

Randomized Search CV was used to tune XGBoost.

### Search Configuration

``` python
RandomizedSearchCV(
    XGB_model,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42
)
```

### 🔧 Search Space

  Parameter         Values
  ----------------- -----------------------
  `n_estimators`    200, 300, 500, 700
  `learning_rate`   0.01, 0.03, 0.05, 0.1
  `max_depth`       3, 4, 5, 6
  `subsample`       0.7, 0.8, 0.9, 1.0

### 🥇 Best Parameters

``` python
{
    'subsample': 0.7,
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.05
}
```

### 🚀 Tuned XGBoost Performance

  Metric               Score
  -------- -----------------
  RMSE       **\$20,502.24**
  MAE        **\$13,817.75**
  R²              **0.9179**

The tuned XGBoost model improved its test RMSE compared with the untuned
XGBoost model, but it still did not outperform the Ridge model on the
final test set.

------------------------------------------------------------------------

## 🏆 Final Model

### Selected Model: Ridge Regression 🧲

Based on the notebook's test-set comparison, **Ridge Regression** was
selected as the best overall model.

#### Why Ridge?

-   ✅ Lowest test RMSE
-   ✅ Lowest test MAE
-   ✅ Highest test R²
-   ✅ Handles correlated predictors through L2 regularization
-   ✅ More interpretable than the tree-based models used in this
    project

### Best Ridge Alpha

Cross-validated alpha:

``` text
alpha = 10.0
```

Search values:

``` python
[0.01, 0.1, 1, 10, 100]
```

------------------------------------------------------------------------

## 🔍 Residual Analysis

Residual analysis was performed for the Ridge model.

### 📊 Residual Distribution

The residuals are roughly centered around zero, but they are not
perfectly normally distributed.

The notebook observed:

-   Most residuals are close to zero.
-   A noticeable right skew exists.
-   There is a long positive tail.
-   Some large residuals remain.

### 📈 Q-Q Plot

The residuals follow the reference line reasonably well around the
center but deviate substantially at both tails.

This indicates that the residuals are only approximately normal and
contain some extreme errors/outliers.

------------------------------------------------------------------------

## 💡 Feature Interpretation

The notebook identified the following as the top five impactful Ridge
features:

    Rank Feature                    Coefficient
  ------ ------------------------ -------------
      1️⃣ `MSZoning_C (all)`             -0.1372
      2️⃣ `Neighborhood_Crawfor`         +0.0704
      3️⃣ `OverallQual`                  +0.0693
      4️⃣ `GrLivArea`                    +0.0684
      5️⃣ `Functional_Typ`               +0.0662

### 🧠 Interpretation

-   `OverallQual` has a positive relationship with the log-transformed
    target.
-   `GrLivArea` has a positive relationship with predicted house price.
-   Neighborhood-related information contributes substantially to price
    prediction.
-   One-hot encoded categorical variables can have their own positive or
    negative coefficients.

------------------------------------------------------------------------

## 📦 Saved Model

The project includes:

``` text
house_price_model.pkl
```

The notebook saves the trained Ridge model using:

``` python
joblib.dump(final_pipeline, 'house_price_model.pkl')
```

The saved object can be loaded using:

``` python
loaded_pipeline = joblib.load('house_price_model.pkl')
```

### ⚠️ Important

The supplied `.pkl` file contains the **Ridge model trained on the
already preprocessed feature matrix**.

The notebook's `final_pipeline` contains only:

``` python
Pipeline([
    ('model', RidgeCV(...))
])
```

Therefore, the preprocessing steps shown in this notebook are performed
**before** the saved model receives the data. A production deployment
would ideally package the preprocessing and model together in a single
end-to-end pipeline.

------------------------------------------------------------------------

## 📁 Project Files

``` text
📦 House Price Prediction
│
├── 📓 House_Price_Prediction.ipynb
├── 🧠 house_price_model.pkl
├── 📊 Ames Housing Dataset.csv
└── 📖 README.md
```

### Files Included

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `House_Price_Prediction.ipynb`      Complete analysis, preprocessing,
                                      modeling, evaluation and
                                      interpretation

  `Ames Housing Dataset.csv`          Dataset used for training and
                                      evaluation

  `house_price_model.pkl`             Saved Ridge regression model

  `README.md`                         Project documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🧰 Technologies Used

### 🐍 Python

Core programming language.

### 🧮 NumPy

Numerical operations and transformations.

### 🐼 Pandas

Data loading, cleaning, manipulation, and analysis.

### 📊 Matplotlib & Seaborn

Visualization and EDA.

### 🔬 SciPy

Statistical analysis, including ANOVA and Q-Q analysis.

### 🤖 Scikit-learn

Used for:

-   Train-test splitting
-   Imputation
-   Encoding
-   Scaling
-   ColumnTransformer
-   Linear Regression
-   RidgeCV
-   LassoCV
-   Random Forest
-   RandomizedSearchCV
-   K-Fold cross-validation
-   Evaluation metrics
-   Pipelines

### 🚀 XGBoost

Gradient-boosted tree regression and hyperparameter tuning.

### 💾 Joblib

Saving and loading the trained model.

------------------------------------------------------------------------

## 📌 Key Learnings

This project demonstrates an end-to-end understanding of a regression
workflow:

``` text
📥 Data Loading
      ↓
🔎 EDA
      ↓
⚠️ Missing-Value Analysis
      ↓
🚨 Outlier Analysis
      ↓
✂️ Train-Test Split
      ↓
🧩 Feature Engineering
      ↓
🔤 Encoding
      ↓
📉 Transformation
      ↓
⚖️ Scaling
      ↓
🤖 Model Training
      ↓
📈 Evaluation
      ↓
🔍 Cross-Validation
      ↓
🎛️ Hyperparameter Tuning
      ↓
🧪 Residual Analysis
      ↓
🏆 Model Selection
      ↓
💾 Model Saving
```

### ⭐ Main Conclusion

Among the models compared in the notebook, **Ridge Regression was the
strongest overall model on the test set**, achieving:

> 🏆 **RMSE: \$19,269.88**\
> 🏆 **MAE: \$13,768.06**\
> 🏆 **R²: 0.9274**

The project therefore demonstrates that a carefully preprocessed and
regularized linear model can outperform the tree-based models tested
here on this particular train-test split.

------------------------------------------------------------------------
