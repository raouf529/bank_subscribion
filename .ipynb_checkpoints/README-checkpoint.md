### bank_subscribion

## description:
A machine learning model to predict whether a client will subscribe to a bank term deposit.
This project was developed as part of the Kaggle Playground Series - S5E8 competition.

## Files Included
- `data/`: Contains the dataset used for training and testing the model.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `model/`: Saved model (.pkl) and metadata (.json).

## Dataset

- **Source:** [Kaggle Playground Series - S5E8: Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8/data)
- **Features:** 27 columns, including:
    - Client demographics: `age`, `job`, `marital`, `education`
    - Financial information: `balance`, `default`, `housing`, `loan`
    - Campaign-related features: `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`
- **Target:** `y` (indicates whether the client subscribed to a term deposit)


## Tools & Libraries

- **Python 3**: Core programming language for the project
- **Scikit-learn**: Machine learning algorithms and utilities
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **XGBoost**: Gradient boosting machine learning library
- **Joblib**: Model serialization and persistence
- **JSON & OS**: Data storage and file system operations

## Data Exploration
- Summary statistics
- Correlation analysis
- Visualizations: histograms, box plots
- Feature importance analysis
- missing value analysis
- outliers handling methods

## split data:
- train_test_split with 80% training and 20% testing  
- random splitting (class distribution stayed nearly the same across train/test sets)  

## missing values:
- `unknown` values in categorical features treated as a separate category
- add `missing_col` feature to indicate missing values for `poutcome` and `contact` 
- no dropping of rows with missing values since missingness is informative

## features engineering:
- add `missing_col` feature to indicate missing values for `poutcome` and `contact` 
- drop `id` feature since it is not useful for prediction

## Preprocessing
### For Linear Models
- Scaling: StandardScaler
- capping + Log transformation where skewness > 1
- One-hot encoding for categorical variables

### For tree-based models
- No scaling/log transform
- One-hot encoding


## Model Selection & Hyperparameter Tuning
- Linear models: Logistic Regression, SGDClassifier
- Tree-based models: decision tree, Random Forest, XGBoost
- RandomizedSearchCV with 5-fold cross-validation

## comparison between models
-  Model performance metrics

| Model | Accuracy | F1 | Precision | Recall | ROC AUC |
|-------|----------|----|-----------|--------|---------|
| Logistic Regression | 0.92 | 0.60 | 0.70 | 0.52 | 0.94 |
| SGDClassifier | 0.91 | 0.58 | 0.72 | 0.49 | 0.94 |
| Decision Tree | 0.91 | 0.51 | 0.73 | 0.39 | 0.89 |
| Random Forest | 0.93 | 0.69 | 0.75 | 0.64 | 0.96 |
| XGBoost | 0.93 | 0.71 | 0.76 | 0.66 | 0.97 |

- Random Forest and XGBoost performed the best among all models.
- XGBoost achieved the highest ROC AUC score, accuracy, and F1 score.
- SGDClassifier and Decision Tree had the lowest performance metrics.
- overall precision is more than recall.


## fine tune and threshold tuning:
- hyperparameter tuning for xgboost using RandomizedSearchCV
- threshold tuning for xgboost, new threshold is 0.35

- after threshold tuning:
| Model | threshold| F1 | Precision | Recall | 
|-------|----------|----|-----------|--------|
| fine tune xgboost | 0.35 | 0.74 | 0.69 | 0.80 |

- after threshold tuning recall increased and precision decreased.

## testing:
- Evaluate the final model on the test set
- Generate classification report, precision-recall curve, roc curve

| Model | threshold| F1 | Precision | Recall | roc_auc |
|-------|----------|----|-----------|--------|---------|
| test set eval | 0.93 | 0.74 | 0.69 | 0.80 | 0.97 |

- The final model demonstrates strong performance on the test set, with a balanced precision and recall.

- precision-recall curve shows the trade-off between precision and recall for different thresholds.
- roc curve illustrates the model's ability to distinguish between classes.

## competition:
- Participate in the Kaggle competition
- Submit the final model's predictions
- Monitor the leaderboard for performance evaluation(0.968)
