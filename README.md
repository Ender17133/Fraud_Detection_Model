# Online Fraud Payment transactions detection model

# Objective: Develop a machine-learning model that will efficiently detect and flag financial fraud transactions. 

## Data used: [link](https://www.kaggle.com/datasets/bannourchaker/frauddetection/data)
**Source: Kaggle**

Synthetic dataset with financial transactions of mobile payments application. 


| Column         | Description                                                          |
|----------------|----------------------------------------------------------------------|
| step           | Represents a unit of time where 1 step equals 1 hour.                |
| type           | The type of online transaction.                                      |
| amount         | The amount of the transaction.                                       |
| nameOrig       | The customer initiating the transaction.                             |
| oldBalanceOrg  | The balance before the transaction of the initiator.                 |
| newBalanceOrig | The balance after the transaction of the initiator.                  |
| nameDest       | The recipient of the transaction.                                    |
| oldBalanceDest | The initial balance of the recipient before the transaction.         |
| newBalanceDest | The new balance of the recipient after the transaction.              |
| isFraud        | Indicates whether the transaction is fraudulent (`Yes` or `No`).     |
| isFlaggedFraud | Column that represent whether Fraud transaction was flagged and stopped correctly by existing measures. |

# Libraries 



```python
# all useful libraries for data analysis

# library that allows importing data from internet sources 
from pathlib import Path

# library that is used for mathematical computations
import numpy as np
from numpy import where 

# library that is used for data management (creating dataframes, aggregating data, cleaning data)
import pandas as pd

# libraries for static data visualizations (Exploratory Data Analysis, Machine Learning (ML) Model evaluation)
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pylab as plt

# logistic regression model from sklearn sklearn library
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# Decision Tree model from sklearn Library
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Decision Tree ensemble models from sklearn library for a higher accuracy and performance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier

# Useful model selection packages from sklearn to optimized model training
## train_test_split - split the data into training and validation (test) samples
## cross_val_score/stratifiedKFold- a quality and reliable way of assessing model performance
## GridSearchCV, RandomizedSearchCV - useful packages that help in search of significant variables for classification
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, StratifiedKFold, GridSearchCV,RandomizedSearchCV


## StandardScaler - useful tool that is used to standardize the data 
## FunctionTransformer - function that allows application of several transformation on the data. 
### For example: log transofrmation
## ColumnTransformer - function that allow additional transformation of columns in the dataset if needed. 
## Make Pipeline - function that allows combining several functions into one object to imrove efficiency of coding.
## sklearn.metrics, scipy.stats, 
### and statsmodels.stats.outliers_influence - useful packages for assessment of data and ML model.
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from scipy.stats import randint
from sklearn.utils import shuffle


## dmba - data mining for Business Analytics package 
import dmba
## dmba packages provides additional methods of assessing ML model and its predictive/classification power. 
from dmba import plotDecisionTree, classificationSummary, gainsChart, liftChart
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba.metric import AIC_score


%matplotlib inline
```

Necessary libraries are imported now. Data should be explored and cleaned if necessary to prepare it for models training. 

# Exploratory Data Analysis (EDA)


```python
# import the data 
## csv from the Kaggle 
df = pd.read_csv('onlinefraud.csv')


```


```python
# read first five rows 
print('First five rows of the data:')
df.head(5)

```

    First five rows of the data:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>isFlaggedFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>9839.64</td>
      <td>C1231006815</td>
      <td>170136.0</td>
      <td>160296.36</td>
      <td>M1979787155</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>1864.28</td>
      <td>C1666544295</td>
      <td>21249.0</td>
      <td>19384.72</td>
      <td>M2044282225</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>181.00</td>
      <td>C1305486145</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>C553264065</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>181.00</td>
      <td>C840083671</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>C38997010</td>
      <td>21182.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>11668.14</td>
      <td>C2048537720</td>
      <td>41554.0</td>
      <td>29885.86</td>
      <td>M1230701703</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# read last five rows 
print('Last five rows of the data:')
df.tail(5)
```

    Last five rows of the data:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>isFlaggedFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6362615</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>339682.13</td>
      <td>C786484425</td>
      <td>339682.13</td>
      <td>0.0</td>
      <td>C776919290</td>
      <td>0.00</td>
      <td>339682.13</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362616</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>6311409.28</td>
      <td>C1529008245</td>
      <td>6311409.28</td>
      <td>0.0</td>
      <td>C1881841831</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362617</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>6311409.28</td>
      <td>C1162922333</td>
      <td>6311409.28</td>
      <td>0.0</td>
      <td>C1365125890</td>
      <td>68488.84</td>
      <td>6379898.11</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362618</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>850002.52</td>
      <td>C1685995037</td>
      <td>850002.52</td>
      <td>0.0</td>
      <td>C2080388513</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362619</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>850002.52</td>
      <td>C1280323807</td>
      <td>850002.52</td>
      <td>0.0</td>
      <td>C873221189</td>
      <td>6510099.11</td>
      <td>7360101.63</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



| Column         | Description                                                          |
|----------------|----------------------------------------------------------------------|
| step           | Represents a unit of time where 1 step equals 1 hour.                |
| type           | The type of online transaction.                                      |
| amount         | The amount of the transaction.                                       |
| nameOrig       | The customer initiating the transaction.                             |
| oldBalanceOrg  | The balance before the transaction of the initiator.                 |
| newBalanceOrig | The balance after the transaction of the initiator.                  |
| nameDest       | The recipient of the transaction.                                    |
| oldBalanceDest | The initial balance of the recipient before the transaction.         |
| newBalanceDest | The new balance of the recipient after the transaction.              |
| isFraud        | Indicates whether the transaction is fraudulent (`Yes` or `No`).     |
| isFlaggedFraud | Column that represent whether Fraud transaction was flagged and stopped correctly by existing measures. |


```python
# information about dataset 
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6362620 entries, 0 to 6362619
    Data columns (total 11 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   step            int64  
     1   type            object 
     2   amount          float64
     3   nameOrig        object 
     4   oldbalanceOrg   float64
     5   newbalanceOrig  float64
     6   nameDest        object 
     7   oldbalanceDest  float64
     8   newbalanceDest  float64
     9   isFraud         int64  
     10  isFlaggedFraud  int64  
    dtypes: float64(5), int64(3), object(3)
    memory usage: 534.0+ MB
    


```python
# Check if data contains any null values 
# dataframe format 
missing_data_df = pd.DataFrame(df.isnull().sum()).reset_index()

# rename columns for clear understanding
missing_data_df.columns = ['column', 'number of rows with missing data']

# check
missing_data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>number of rows with missing data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>step</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>type</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>amount</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>nameOrig</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>oldbalanceOrg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>newbalanceOrig</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nameDest</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>oldbalanceDest</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>newbalanceDest</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>isFraud</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>isFlaggedFraud</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



There are no missing rows in the data, I can proceed with EDA. 

## Effectiveness of existing Anti-Fraud measures 


There is a `isFlaggedFraud` variable, which could be interpreted as if there already is some kind of algoritm that detects fraud transactions. It would be wise to validate the accuracy of this "algorithm".


```python
# data to show how many fraud transactions were flagged correctly 

# all fraud transactions 
all_fraud = df[df['isFraud'] == 1].shape[0]

# not flagged but fraud
not_flagged_fraud = df[(df['isFraud'] == 1) & (df['isFlaggedFraud'] == 0)].shape[0]

# flagged and fraud
flagged_fraud = df[(df['isFraud'] == 1) & (df['isFlaggedFraud'] == 1)].shape[0]

# fraud 
fraud = [not_flagged_fraud, flagged_fraud]

# percentage not flagged but fraud 
not_flagged_fraud_pct = (not_flagged_fraud / all_fraud) * 100

# percentage flagged and fraud 
flagged_fraud_pct = (flagged_fraud / all_fraud) * 100
```


```python
# pie-chart plot 
# plot size 
# 16 - length 
# 10 - height 
plt.figure(figsize = (16, 10))

# function to show labels and percentages of fraud transactions for each type 
labels_with_info = ['{}: {:.2f}%\n({})'.format(types, percentage, counts) for types, percentage, counts in 
                    zip(['0 (Not Flagged but Fraud', '1 (Flagged and Fraud)'], 
                        [not_flagged_fraud_pct, flagged_fraud_pct], 
                        fraud
)]
 
# pie-chart plot 
plt.pie( 
    # values
    fraud,
    # labels
    labels = labels_with_info,
    # text size 
    textprops = {'fontsize': 18}, 
    # color palette
    colors = ['#66c2a5', '#e78ac3'])

# circle at the center of pie to make it a donut 
circle = plt.Circle((0, 0), 0.80, fc = 'white')
plt.gcf().gca().add_artist(circle)

# add total in the center 
total = (not_flagged_fraud + flagged_fraud)

# apply total number on pie chart 
plt.text(0, 0, f"Total number of Fraud Transactions {total}", 
               ha='center', 
               va='center', fontsize=16)

# parameters
## title 
plt.title('Effectiveness of existing Anti-Fraud measures', fontweight = 'bold', fontsize = 18)

plt.show()


```


    
![png](output_15_0.png)
    


Among 8213 transactions, only 16 fraud transactions were effectively flagged, which indicates that there is a need of a new algorithm that will effectively recognize fraud transactions. 

## Transaction types 



```python
# check types of the transactions and count values of transactions
# dataframe format
transaction_types_df = pd.DataFrame(df.type.value_counts()).reset_index()

# rename columns for clear understanding
transaction_types_df.columns = ['transaction_type', 'transactions_number']

# total number of transactions
total_number = transaction_types_df['transactions_number'].sum()

# percentage of each transaction
transaction_types_df['percentage'] = (transaction_types_df['transactions_number'] / total_number) * 100

# check 
transaction_types_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction_type</th>
      <th>transactions_number</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CASH_OUT</td>
      <td>2237500</td>
      <td>35.166331</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PAYMENT</td>
      <td>2151495</td>
      <td>33.814608</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH_IN</td>
      <td>1399284</td>
      <td>21.992261</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRANSFER</td>
      <td>532909</td>
      <td>8.375622</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DEBIT</td>
      <td>41432</td>
      <td>0.651178</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check number of fraud transactions by each type of transaction
# fraud transaction = 1
fraud_transactions = df.groupby('type')['isFraud'].sum().reset_index()
    
# rename columns 
fraud_transactions.columns = ['transaction_type', 'fraud_transactions_number']

# sort from biggest number to the least 
fraud_transactions = fraud_transactions.sort_values('fraud_transactions_number', ascending = False)

# show overall graph 
# check 
fraud_transactions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction_type</th>
      <th>fraud_transactions_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>CASH_OUT</td>
      <td>4116</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRANSFER</td>
      <td>4097</td>
    </tr>
    <tr>
      <th>0</th>
      <td>CASH_IN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DEBIT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PAYMENT</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge two dataframes 
transaction_types =  pd.merge(transaction_types_df, 
                             fraud_transactions, 
                             on = 'transaction_type', 
                             how = 'left')

# create a variable that represents percentage of fraud transaction among all transactions 
computation = (transaction_types['fraud_transactions_number'] / transaction_types['transactions_number']) * 100
transaction_types['fraud_percentage'] = computation

# sort 
# sort from biggest number to the least 
transaction_types = transaction_types.sort_values('fraud_transactions_number', ascending = False)

# create a dataframe for plot 
## filter 
filter_row = transaction_types['fraud_transactions_number'] != 0
## apply filter 
transaction_types_plot = transaction_types.loc[filter_row, :]

# check 
transaction_types




```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction_type</th>
      <th>transactions_number</th>
      <th>percentage</th>
      <th>fraud_transactions_number</th>
      <th>fraud_percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CASH_OUT</td>
      <td>2237500</td>
      <td>35.166331</td>
      <td>4116</td>
      <td>0.183955</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRANSFER</td>
      <td>532909</td>
      <td>8.375622</td>
      <td>4097</td>
      <td>0.768799</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PAYMENT</td>
      <td>2151495</td>
      <td>33.814608</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH_IN</td>
      <td>1399284</td>
      <td>21.992261</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DEBIT</td>
      <td>41432</td>
      <td>0.651178</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# function to show labels and percentages of fraud transactions for each type 
labels_with_info = ['{}: {:.2f}%\n({})'.format(types, percentage, counts) for types, percentage, counts in 
                    zip(transaction_types_plot['transaction_type'], transaction_types_plot['fraud_percentage'], 
                        transaction_types_plot['fraud_transactions_number']
)]


# subplot 1 row, 2 columns, 1st figure 
plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 1) 
# bar-chart code 
barplot = sns.barplot(# data
            data = transaction_types_df,
            # x
            x = 'transactions_number', 
            # y
            y = 'transaction_type', 
            # color palette
            palette = sns.color_palette('Set2'))


# parameters 
# x-label 
plt.xlabel('Number of transactions (units)', fontsize = 16)
plt.xticks(fontsize = 16)
# y-label 
plt.ylabel('Transaction type', fontsize = 16)
plt.yticks(fontsize = 16)
# title 
plt.title('Number of customer transactions by type of the transaction', fontweight = 'bold', fontsize = 18)

# add annotations
for index, row in transaction_types_df.iterrows():
    # horizontal alignment 
    if index in transaction_types_df.index[-2:]:
        ha_alignment = 'left'
    
    else:
        ha_alignment = 'right'
    
    # apply text on barplot
    barplot.text(row['transactions_number'],index, f"{row['transactions_number']} ({row['percentage']:.2f})%", 
            color = 'black', ha = ha_alignment, fontsize = 18)




# suplot 1 row, 2 columns, 2nd figure 
plt.subplot(1, 2, 2)

# pie-chart plot 
plt.pie(
    # values
    transaction_types_plot['fraud_transactions_number'],
    # labels
    labels = labels_with_info,
    # text size 
    textprops = {'fontsize': 18}, 
    # color palette
    colors = ['#66c2a5', '#e78ac3'])

# add total in the center 
total = (np.sum(transaction_types['fraud_transactions_number']) / np.sum(transaction_types['transactions_number'])) * 100

# apply total number on pie chart 
plt.text(0, 0, f"Total % of fraud transactions among all transactions:\n{round(total, 3)}%", 
               ha='center', 
               va='center', fontsize=16)

# title 
plt.title('Number of fraud transactions by type of the transaction', fontweight = 'bold', fontsize = 18)

# show plot
plt.tight_layout()
plt.show()

```


    
![png](output_21_0.png)
    


`Cashout` and `Payment` transactions are the most common and popular transactions among customers. However, data also suggests that fraud cases are only present in `Cashout` and less popular `Transfer` transactions. 

## Financial values of each transaction type and potential financial costs (fraud transactions)



```python
# dataframe : median customer transaction amount for each transaction type 
transaction_amount = df[df['isFraud'] != 1].groupby('type')['amount'].median().reset_index()

# rename columns 
transaction_amount.columns = ['transaction_type', 'median_amount_usd']

# sort values 
transaction_amount = transaction_amount.sort_values('median_amount_usd', ascending = False)

# check 
transaction_amount
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction_type</th>
      <th>median_amount_usd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>TRANSFER</td>
      <td>486521.91</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CASH_OUT</td>
      <td>146946.56</td>
    </tr>
    <tr>
      <th>0</th>
      <td>CASH_IN</td>
      <td>143427.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PAYMENT</td>
      <td>9482.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DEBIT</td>
      <td>3048.99</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dataframe: median fraud transaction amount for each transaction type 
fraud_amount = df[df['isFraud'] == 1].groupby('type')['amount'].median().reset_index()

# rename columns 
fraud_amount.columns = ['transaction_type', 'median_fraud_amount_usd']

# check 
fraud_amount
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction_type</th>
      <th>median_fraud_amount_usd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CASH_OUT</td>
      <td>435516.905</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRANSFER</td>
      <td>445705.760</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge dataframes 
transaction_amount = pd.merge(transaction_amount, 
                             fraud_amount, 
                             on = 'transaction_type', 
                             how = 'left')

# check 
transaction_amount
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transaction_type</th>
      <th>median_amount_usd</th>
      <th>median_fraud_amount_usd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRANSFER</td>
      <td>486521.91</td>
      <td>445705.760</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CASH_OUT</td>
      <td>146946.56</td>
      <td>435516.905</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH_IN</td>
      <td>143427.71</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PAYMENT</td>
      <td>9482.19</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DEBIT</td>
      <td>3048.99</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# bar-charts plot 
# plot size 
# 20 - length 
# 10 - height 
plt.figure(figsize = (20, 10))

# bar width 
bar_width = 0.30

# positions for set of bars 
positions = np.arange(len(transaction_amount['transaction_type']))

# median transaction amount bar 
bar1 = plt.bar(
    # x 
    positions, 
    # y
    transaction_amount['median_amount_usd'],
    # bar width 
    width = bar_width, 
    # label
    label  = 'Median customer transaction amount ($)')

# median fraud amount bar 
bar2 = plt.bar(
    # x
    positions + bar_width, 
    # y
    transaction_amount['median_fraud_amount_usd'], 
    # bar width 
    width = bar_width, 
    # label 
    label = 'Median fraud transaction amount ($)')


# add annotations
for index, row in transaction_amount.iterrows():
    # apply text on both barplots
    # customer transactions barplot
    plt.text(index, row['median_amount_usd'] + 5, f"{int(row['median_amount_usd']):,}$", 
             # color of text
             color = 'black', 
             # horizontal alignment 
             ha = 'center', 
             # vertical alignment
             va = 'baseline', 
            # font size 
             fontsize = 16)
    
    # fraud transactions barplot 
    # if fraud value is not missing 
    if not pd.isnull(row['median_fraud_amount_usd']):
        plt.text(index + bar_width, 
             row['median_fraud_amount_usd'] + 5, 
             f"{int(row['median_fraud_amount_usd']):,}$", 
             color='black', 
             ha='center',
             va = 'baseline', 
             fontsize=16)

    


# parameters
## title 
plt.title('Median Transaction amount in USD by Transaction type', fontweight = 'bold', fontsize = 18)

## xlabel
plt.xlabel('Transaction type', fontsize = 16)

## x-axis 
plt.xticks(positions + bar_width / 2, transaction_amount['transaction_type'].to_list(), fontsize = 16)

## ylabel
plt.ylabel('Median Transaction amount ($)', fontsize = 16)

## y-axis 
plt.yticks(fontsize = 16)

# legend
plt.legend(loc = 'upper right', fontsize = 18)


plt.show()


```


    
![png](output_27_0.png)
    


Graph above indicates that fraud transactions have very high financial costs for the firm. Median amount of fraud transactions in TRANSFER transactions is nearlly equal to customer amount and median fraud transaction amoun in CASH_OUT transactions is 4 times higher.


```python
# library that is needed now 
import matplotlib.ticker as ticker

# datafrmame `transactions` to show money distribution of fraud transactions
transactions_fraud = df[df.isFraud == 1]
transactions_fraud = transactions_fraud.sort_values(by = ['amount'], ascending = False)



# 1st plot 
# histogram plot
transactions_fraud.amount.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # plot size (length, height)
    figsize = (16, 10), 
    # bin edge color
    edgecolor = 'black')

# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title('Distribution of Online Fraudulent Transaction Amounts', fontsize = 18, fontweight = 'bold')
## xlabel
plt.xlabel('Fraudulent Transaction Amounts', fontsize = 16)
## xticks 
plt.xticks(fontsize = 16)
## ylabel
plt.ylabel('Frequency', fontsize = 16)
## yticks
plt.yticks(fontsize = 16)




```




    (array([   0., 1000., 2000., 3000., 4000., 5000., 6000.]),
     [Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, '')])




    
![png](output_29_1.png)
    


Most of the fraud transactions amounts are up to 4 million dollars, but there are transactions present which reach to 10 millions dollars. This insight again indicates that company is suffering from high financial losses and strong machine-learning algorithm is needed. 

## Customer Transaction Origin account


```python
# new dataframe with only customer cases 
accounts = df[df['isFraud'] == 0]

# how many times similar account was used for fraud
customer = accounts['nameOrig'].value_counts()

# top 10 rows of fraud dataframe 
top10_customer = customer.head(10)

# bottom 10 rows of fraud dataframe
bottom10_customer = customer.tail(10)

# merge 2 subsets 
merged_customer = pd.concat([top10_customer, bottom10_customer])

# convert to dataframe 
merged_customer = pd.DataFrame(merged_customer).reset_index()

# rename columns 
merged_customer.columns = ['original_account_number', 'times_used']

merged_customer
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original_account_number</th>
      <th>times_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C1902386530</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C2051359467</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C1677795071</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C400299098</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C363736674</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C1832548028</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C1784010646</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C1530544995</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C2098525306</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C1976208114</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>C1653939292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>C1944988793</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>C486054054</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>C949435242</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>C1624122524</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>C638139918</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>C393639841</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>C1800825581</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>C1040150492</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>C49652609</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# bar plot showing top 10 and bottom 10 customer accounts
# figure size 
plt.figure(figsize = (16, 10))


# bar plot 
sns.barplot( 
       # x
       merged_customer['times_used'], 
       # y 
       merged_customer['original_account_number'],
    # color 
    color = 'skyblue')


# parameters 
## title 
plt.title('Top 10 and bottom 10 sender customer accounts', fontsize = 18, fontweight = 'bold')

## xlabel 
plt.xlabel('Number of times account was used for transaction', fontsize = 16)
## xaxis 
plt.xticks(fontsize = 16)

## ylabel 
plt.ylabel('Account numbers', fontsize = 16)
## yaxis 
plt.yticks(fontsize = 16)
```

    C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19]),
     [Text(0, 0, 'C1902386530'),
      Text(0, 1, 'C2051359467'),
      Text(0, 2, 'C1677795071'),
      Text(0, 3, 'C400299098'),
      Text(0, 4, 'C363736674'),
      Text(0, 5, 'C1832548028'),
      Text(0, 6, 'C1784010646'),
      Text(0, 7, 'C1530544995'),
      Text(0, 8, 'C2098525306'),
      Text(0, 9, 'C1976208114'),
      Text(0, 10, 'C1653939292'),
      Text(0, 11, 'C1944988793'),
      Text(0, 12, 'C486054054'),
      Text(0, 13, 'C949435242'),
      Text(0, 14, 'C1624122524'),
      Text(0, 15, 'C638139918'),
      Text(0, 16, 'C393639841'),
      Text(0, 17, 'C1800825581'),
      Text(0, 18, 'C1040150492'),
      Text(0, 19, 'C49652609')])




    
![png](output_33_2.png)
    


## Customer and Fraud Transaction Destination Accounts


```python
# new dataframe with only customer recipients (not fraud) 
accounts = df[df['isFraud'] == 0]

# how many times similar account was used for fraud
customer = accounts['nameDest'].value_counts()

# top 10 rows of fraud dataframe 
top10_customer = customer.head(10)

# bottom 10 rows of fraud dataframe
bottom10_customer = customer.tail(10)

# merge 2 subsets 
merged_customer = pd.concat([top10_customer, bottom10_customer])

# convert to dataframe 
merged_customer = pd.DataFrame(merged_customer).reset_index()

# rename columns 
merged_customer.columns = ['destination_account_number', 'times_used']

merged_customer

# C - Customers 
# M - Merchants (Sellers)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>destination_account_number</th>
      <th>times_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C1286084959</td>
      <td>113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C985934102</td>
      <td>109</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C665576141</td>
      <td>105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C2083562754</td>
      <td>102</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C248609774</td>
      <td>101</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C1590550415</td>
      <td>101</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C451111351</td>
      <td>99</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C1789550256</td>
      <td>99</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C1360767589</td>
      <td>98</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C1023714065</td>
      <td>97</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M670653449</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1286010320</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M1993825937</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M2005237610</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M1425965481</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M435905844</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M534128698</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M2140757139</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M316120022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>C855350324</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# new dataframe with only fraud cases 
accounts = df[df['isFraud'] == 1]

# how many times similar account was used for fraud
fraud = accounts['nameDest'].value_counts()

# top 10 rows of fraud dataframe 
top10_fraud = fraud.head(10)

# bottom 10 rows of fraud dataframe
bottom10_fraud = fraud.tail(10)

# merge 2 subsets 
merged_fraud = pd.concat([top10_fraud, bottom10_fraud])

# convert to dataframe 
merged_fraud = pd.DataFrame(merged_fraud).reset_index()

# rename columns 
merged_fraud.columns = ['destination_account_number', 'times_used']

# check 
merged_fraud
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>destination_account_number</th>
      <th>times_used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C1193568854</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C104038589</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C200064275</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C1497532505</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C1601170327</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C1655359478</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C2020337583</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C1653587362</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C1013511446</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>C2129197098</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>C1955464150</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>C1692585098</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>C1566713324</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>C1704299654</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>C1959131299</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>C317811789</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>C24324787</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>C1053414206</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>C2013070624</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>C873221189</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# figure size 
plt.figure(figsize = (16, 15))

# plot 1 
plt.subplot(2, 1, 1)
# bar plot showing top 10 and bottom 10 customer recipient accounts 
# bar plot 
sns.barplot( 
       # x
       merged_customer['times_used'], 
       # y 
       merged_customer['destination_account_number'],
    # color 
    color = 'skyblue')


# parameters 
## title 
plt.title('Top 10 and bottom 10 recipient accounts used in customer transactions', fontsize = 18, fontweight = 'bold')

## xlabel 
plt.xlabel('Number of times account was used for transaction', fontsize = 16)
## xaxis 
plt.xticks(fontsize = 16)

## ylabel 
plt.ylabel('Account numbers', fontsize = 16)
## yaxis 
plt.yticks(fontsize = 16)
## legend
plt.legend(['M: Merchant', 
            "C: Customer"], 
          fontsize = 16, 
          loc = 'lower right')

# plot 2 
plt.subplot(2, 1, 2)

# bar plot showing top 10 and bottom 10 fraud recipient accounts 
# bar plot 
# bar plot showing top 10 and bottom 10 fraud accounts



# bar plot 
sns.barplot( 
       # x
       merged_fraud['times_used'], 
       # y 
       merged_fraud['destination_account_number'],
    # color 
    color = 'skyblue')


# parameters 
## title 
plt.title('Top 10 and bottom 10 recipient accounts used in fraud transactions', fontsize = 18, fontweight = 'bold')

## xlabel 
plt.xlabel('Number of times account was used for transaction', fontsize = 16)
## xaxis 
plt.xticks(fontsize = 16)

## ylabel 
plt.ylabel('Account numbers', fontsize = 16)
## yaxis 
plt.yticks(fontsize = 16)

plt.show()
```

    C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_37_1.png)
    


Very useful insight - "Merchant" accounts were never used in a fraud transaction. That could be a business rule for the model: if the recepient in transaction is a "Merchant" account, such transaction should not be flagged and stopped, unless there are factors like: unusual geolocation, suspicious activities, which I don't cover in this project. 

## Distribution of old and new accounts balances during fraud `Cash Out` transactions


```python
# new dataframe that has old and new balances with only `Cash Out` transactions
sender_fraud_cashout = df[(df['isFraud'] == 1) & (df['type'] == 'CASH_OUT')]

# check 
sender_fraud_cashout
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>isFlaggedFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>181.00</td>
      <td>C840083671</td>
      <td>181.00</td>
      <td>0.0</td>
      <td>C38997010</td>
      <td>21182.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>252</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>2806.00</td>
      <td>C2101527076</td>
      <td>2806.00</td>
      <td>0.0</td>
      <td>C1007251739</td>
      <td>26202.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>681</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>20128.00</td>
      <td>C1118430673</td>
      <td>20128.00</td>
      <td>0.0</td>
      <td>C339924917</td>
      <td>6268.00</td>
      <td>12145.85</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>724</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>416001.33</td>
      <td>C749981943</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C667346055</td>
      <td>102.00</td>
      <td>9291619.62</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>970</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>1277212.77</td>
      <td>C467632528</td>
      <td>1277212.77</td>
      <td>0.0</td>
      <td>C716083600</td>
      <td>0.00</td>
      <td>2444985.19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6362611</th>
      <td>742</td>
      <td>CASH_OUT</td>
      <td>63416.99</td>
      <td>C994950684</td>
      <td>63416.99</td>
      <td>0.0</td>
      <td>C1662241365</td>
      <td>276433.18</td>
      <td>339850.17</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362613</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>1258818.82</td>
      <td>C1436118706</td>
      <td>1258818.82</td>
      <td>0.0</td>
      <td>C1240760502</td>
      <td>503464.50</td>
      <td>1762283.33</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362615</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>339682.13</td>
      <td>C786484425</td>
      <td>339682.13</td>
      <td>0.0</td>
      <td>C776919290</td>
      <td>0.00</td>
      <td>339682.13</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362617</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>6311409.28</td>
      <td>C1162922333</td>
      <td>6311409.28</td>
      <td>0.0</td>
      <td>C1365125890</td>
      <td>68488.84</td>
      <td>6379898.11</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362619</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>850002.52</td>
      <td>C1280323807</td>
      <td>850002.52</td>
      <td>0.0</td>
      <td>C873221189</td>
      <td>6510099.11</td>
      <td>7360101.63</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4116 rows × 11 columns</p>
</div>




```python
# graph size 
# length, height
plt.figure(figsize = (28, 10))
# 1st graph - distribution of old and new balance amounts of sender during the fraud cashout transactions
# 1st plot - sender account balances before Fraud transaction
plt.subplot(1, 2, 1)
# histogram plot
sender_fraud_cashout.oldbalanceOrg.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black', 
     # range
    range = (0, 10000000)
)
# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Sender account balances before Fraud `Cash Out` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('Old balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)

#2nd plot - sender account balances after Fraud transaction
plt.subplot(1, 2, 2)
# histogram plot
sender_fraud_cashout.newbalanceOrig.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black',
     # range
   range = (0, 10000000)
)

# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Sender account balances after Fraud `Cash Out` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('New balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)

plt.show()
```


    
![png](output_41_0.png)
    



```python
# graph size 
# length, height
plt.figure(figsize = (28, 10))
# 2nd graph - distribution of old and new balance amounts of fraud recipient accounts during the fraud cashout transactions
# 1st plot - fraud recipient account balances before Fraud transaction
plt.subplot(1, 2, 1)
# histogram plot
sender_fraud_cashout.oldbalanceDest.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black',
     # range
    range = (0, 11000000)
)
# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Fraud recipient account balances before Fraud `Cash Out` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('Old balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)

#2nd plot - fraud recipient account balances after Fraud transaction
plt.subplot(1, 2, 2)
# histogram plot
sender_fraud_cashout.newbalanceDest.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black', 
     # range
   range = (0, 11000000)
)

# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Fraud recipient account balances after Fraud `Cash Out` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('New balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)
```




    (array([   0.,  200.,  400.,  600.,  800., 1000., 1200., 1400., 1600.,
            1800.]),
     [Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, '')])




    
![png](output_42_1.png)
    


## Distribution of old and new accounts balances during fraud `TRANSFER` transactions


```python
# new dataframe that has old and new balances with only `TRANSFER` transactions
sender_fraud_transfer = df[(df['isFraud'] == 1) & (df['type'] == 'TRANSFER')]

# check 
sender_fraud_transfer
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>isFlaggedFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>181.00</td>
      <td>C1305486145</td>
      <td>181.00</td>
      <td>0.0</td>
      <td>C553264065</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>251</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>2806.00</td>
      <td>C1420196421</td>
      <td>2806.00</td>
      <td>0.0</td>
      <td>C972765878</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>680</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>20128.00</td>
      <td>C137533655</td>
      <td>20128.00</td>
      <td>0.0</td>
      <td>C1848415041</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>969</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>1277212.77</td>
      <td>C1334405552</td>
      <td>1277212.77</td>
      <td>0.0</td>
      <td>C431687661</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1115</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>35063.63</td>
      <td>C1364127192</td>
      <td>35063.63</td>
      <td>0.0</td>
      <td>C1136419747</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6362610</th>
      <td>742</td>
      <td>TRANSFER</td>
      <td>63416.99</td>
      <td>C778071008</td>
      <td>63416.99</td>
      <td>0.0</td>
      <td>C1812552860</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362612</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>1258818.82</td>
      <td>C1531301470</td>
      <td>1258818.82</td>
      <td>0.0</td>
      <td>C1470998563</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362614</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>339682.13</td>
      <td>C2013999242</td>
      <td>339682.13</td>
      <td>0.0</td>
      <td>C1850423904</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362616</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>6311409.28</td>
      <td>C1529008245</td>
      <td>6311409.28</td>
      <td>0.0</td>
      <td>C1881841831</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362618</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>850002.52</td>
      <td>C1685995037</td>
      <td>850002.52</td>
      <td>0.0</td>
      <td>C2080388513</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4097 rows × 11 columns</p>
</div>




```python
# graph size 
# length, height
plt.figure(figsize = (28, 10))
# 1st graph - distribution of old and new balance amounts of sender during the fraud transfer transactions
# 1st plot - sender account balances before Fraud transaction
plt.subplot(1, 2, 1)
# histogram plot
sender_fraud_transfer.oldbalanceOrg.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black', 
     # range
    range = (0, 10000000)
)
# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Sender account balances before Fraud `Transfer` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('Old balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)

#2nd plot - sender account balances after Fraud transaction
plt.subplot(1, 2, 2)
# histogram plot
sender_fraud_transfer.newbalanceOrig.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black',
     # range
   range = (0, 10000000)
)

# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Sender account balances after Fraud `Transfer` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('New balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)
```




    (array([   0.,  500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.,
            4500.]),
     [Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, '')])




    
![png](output_45_1.png)
    



```python
# graph size 
# length, height
plt.figure(figsize = (28, 10))
# 2nd graph - distribution of old and new balance amounts of fraudulent recipient during the fraud transfer transactions
# 1st plot - fraudulent account balances before Fraud transaction
plt.subplot(1, 2, 1)
# histogram plot
sender_fraud_transfer.oldbalanceDest.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black'
     # range
   #range = (0, 1000000)
)
# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Fraud account balances before Fraud `Transfer` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('Old balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)

#2nd plot - fraudulent account balances after Fraud transaction
plt.subplot(1, 2, 2)
# histogram plot
sender_fraud_transfer.newbalanceDest.plot(
    # figure type 
    kind = 'hist', 
    # number of bins
    bins = 15,
    # bin color
    facecolor = 'orange',
    # bin edge color
    edgecolor = 'black'
     # range
  #range = (0, 1000000)
)

# format of x axist to show full values
plt.ticklabel_format(style='plain', axis='x')  

# show values through coma
formats = ticker.StrMethodFormatter('{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formats)

# parameters 
## title
plt.title("Fraud account balances after Fraud `Transfer` transaction", fontsize = 20, fontweight = 'bold')
## xlabel
plt.xlabel('New balance', fontsize = 18)
## xticks 
plt.xticks(fontsize = 18)
## ylabel
plt.ylabel('Frequency', fontsize = 18)
## yticks
plt.yticks(fontsize = 18)
```




    (array([   0.,  500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.,
            4500.]),
     [Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, ''),
      Text(0, 0, '')])




    
![png](output_46_1.png)
    


Interesting question that needs to be answered is "How much money would have been saved if the fraud transactions had been detected and prevented earlier?"


```python
# variable with only fraud transactions 
fraud_transactions = df[(df['isFraud'] == 1) & (df['isFlaggedFraud'] == 0)]

# total amount
total_fraud_amount = fraud_transactions['amount'].sum()

print(f"Total amount of money that would have been saved: {total_fraud_amount}$")
```

    Total amount of money that would have been saved: 11978629864.15$
    

# Data cleaning 

## Remove uneccesary columns 



```python
# information about the main variable with data 
df.info()

# columns that can't be used for classification algorithm training: 
# step - good for additional data visuzalition, but can't be used for predictions.

# nameOrig - can't be used for predictions as well,
## but provide useful information about accounts used. 

# isFlaggedFraud - explains how accurately did previous algorithm flag fraud transactions, shouldn't be used for 
## prediction as it will cause a data leakage. 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6362620 entries, 0 to 6362619
    Data columns (total 11 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   step            int64  
     1   type            object 
     2   amount          float64
     3   nameOrig        object 
     4   oldbalanceOrg   float64
     5   newbalanceOrig  float64
     6   nameDest        object 
     7   oldbalanceDest  float64
     8   newbalanceDest  float64
     9   isFraud         int64  
     10  isFlaggedFraud  int64  
    dtypes: float64(5), int64(3), object(3)
    memory usage: 534.0+ MB
    


```python
# create a variable that will be used for the rest of classification training
# copy df 
df_model = df.copy()

# list with columns that need to be removed 
remove_columns = ['step', 'nameOrig', 'isFlaggedFraud']

# remove columns 
df_model.drop(columns = remove_columns, axis = 1, inplace = True)
```

## Removing uneccesary rows

In this data, there might be one logical error which could lead to confusing predictions and outcomes - `transaction amount higher than the balance amount possible`. If exclude such factors as credits, over-limit pay, this case is not logical and relevant for fraud detection.


```python
# check how many rows are there with amount higher than balance
rows = df_model[df_model['amount'] > df_model['oldbalanceOrg']]

# check 
rows.shape[0]
```




    4079080




```python
# check how many fraud transactions are there
rows_fraud = rows[rows['isFraud'] == 1]

# check 
rows_fraud.shape[0]
```




    29




```python
# show these fraud rows 
rows_fraud
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>724</th>
      <td>CASH_OUT</td>
      <td>416001.33</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C667346055</td>
      <td>102.00</td>
      <td>9291619.62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>CASH_OUT</td>
      <td>132842.64</td>
      <td>4499.08</td>
      <td>0.0</td>
      <td>C297927961</td>
      <td>0.00</td>
      <td>132842.64</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14861</th>
      <td>CASH_OUT</td>
      <td>181728.11</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C789014007</td>
      <td>11397.00</td>
      <td>184477.77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25875</th>
      <td>TRANSFER</td>
      <td>1078013.76</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C277510102</td>
      <td>0.00</td>
      <td>970749.68</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77745</th>
      <td>CASH_OUT</td>
      <td>277970.88</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C571514738</td>
      <td>0.00</td>
      <td>277970.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>138559</th>
      <td>TRANSFER</td>
      <td>1933920.80</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C461905695</td>
      <td>1283762.85</td>
      <td>3217683.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>169998</th>
      <td>CASH_OUT</td>
      <td>149668.66</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C460735540</td>
      <td>44170.11</td>
      <td>193838.76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>178668</th>
      <td>CASH_OUT</td>
      <td>222048.71</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1700442291</td>
      <td>2979.00</td>
      <td>225027.71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>200845</th>
      <td>CASH_OUT</td>
      <td>454859.39</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C2146670328</td>
      <td>0.00</td>
      <td>454859.39</td>
      <td>1</td>
    </tr>
    <tr>
      <th>217978</th>
      <td>TRANSFER</td>
      <td>123194.95</td>
      <td>79466.45</td>
      <td>0.0</td>
      <td>C1755380031</td>
      <td>535933.16</td>
      <td>263908.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>291459</th>
      <td>CASH_OUT</td>
      <td>95428.32</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1720721903</td>
      <td>0.00</td>
      <td>95428.32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>296686</th>
      <td>CASH_OUT</td>
      <td>39713.28</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1795377601</td>
      <td>1274866.51</td>
      <td>1314579.79</td>
      <td>1</td>
    </tr>
    <tr>
      <th>408955</th>
      <td>CASH_OUT</td>
      <td>314251.58</td>
      <td>75956.47</td>
      <td>0.0</td>
      <td>C90486891</td>
      <td>7962205.25</td>
      <td>8276456.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>424928</th>
      <td>CASH_OUT</td>
      <td>508782.20</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C2090737806</td>
      <td>1082007.65</td>
      <td>1590789.85</td>
      <td>1</td>
    </tr>
    <tr>
      <th>479636</th>
      <td>CASH_OUT</td>
      <td>122101.57</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1200316948</td>
      <td>0.00</td>
      <td>639940.20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>543928</th>
      <td>CASH_OUT</td>
      <td>23292.30</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1834461593</td>
      <td>392364.62</td>
      <td>415656.92</td>
      <td>1</td>
    </tr>
    <tr>
      <th>559979</th>
      <td>CASH_OUT</td>
      <td>89571.46</td>
      <td>4505.60</td>
      <td>0.0</td>
      <td>C1460548505</td>
      <td>1929428.01</td>
      <td>2018999.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>643671</th>
      <td>CASH_OUT</td>
      <td>112280.88</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C708118422</td>
      <td>40512.49</td>
      <td>152793.36</td>
      <td>1</td>
    </tr>
    <tr>
      <th>694551</th>
      <td>CASH_OUT</td>
      <td>234377.29</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C877378703</td>
      <td>34937.86</td>
      <td>269315.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>732891</th>
      <td>CASH_OUT</td>
      <td>112486.46</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C179706450</td>
      <td>257274.47</td>
      <td>369760.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>750755</th>
      <td>CASH_OUT</td>
      <td>577418.98</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C541373010</td>
      <td>0.00</td>
      <td>577418.98</td>
      <td>1</td>
    </tr>
    <tr>
      <th>764187</th>
      <td>CASH_OUT</td>
      <td>407005.78</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1059308371</td>
      <td>0.00</td>
      <td>407005.78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>920521</th>
      <td>TRANSFER</td>
      <td>1395850.55</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1429415136</td>
      <td>260806.21</td>
      <td>1656656.77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1021951</th>
      <td>TRANSFER</td>
      <td>202978.65</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C966173999</td>
      <td>2122336.55</td>
      <td>2325315.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2058343</th>
      <td>CASH_OUT</td>
      <td>332729.54</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1797918851</td>
      <td>613712.35</td>
      <td>946441.90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2242699</th>
      <td>CASH_OUT</td>
      <td>229909.57</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1620159110</td>
      <td>0.00</td>
      <td>229909.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2622102</th>
      <td>CASH_OUT</td>
      <td>291519.84</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C1138105020</td>
      <td>0.00</td>
      <td>291519.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2946481</th>
      <td>CASH_OUT</td>
      <td>40611.22</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C478307499</td>
      <td>0.00</td>
      <td>40611.22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2983493</th>
      <td>CASH_OUT</td>
      <td>94372.61</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>C421443093</td>
      <td>471783.48</td>
      <td>566156.08</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



There are ~4 million rows and from these 4 millions, there are only 29 fraud transactions where the amount of transaction is higher than the balance. However, after the exploration it actually might make sense. 
1) Fraudster focused on an account with 0 balance

2) If there is some balance on consumer account, such fraudster transaction resulted in full depletion of balance reserves. 

To be safe, I am going to delete such non-fraud rows, while leaving 29 fraud rows in the data. 


```python
# clean 
df_model = df_model[~((df_model['amount'] > df_model['oldbalanceOrg']))]

```


```python
# check whether worked 
df_model.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2283540 entries, 0 to 6362619
    Data columns (total 8 columns):
     #   Column          Dtype  
    ---  ------          -----  
     0   type            object 
     1   amount          float64
     2   oldbalanceOrg   float64
     3   newbalanceOrig  float64
     4   nameDest        object 
     5   oldbalanceDest  float64
     6   newbalanceDest  float64
     7   isFraud         int64  
    dtypes: float64(5), int64(1), object(2)
    memory usage: 156.8+ MB
    


```python
# check whether there is still these 29 rows 
df_model[(df_model['amount'] > df_model['oldbalanceOrg']) & df_model['isFraud'] == 1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Other types of data cleaning such as converting to dummy variables, z-normalization, or outliers can't be done before data partitioning, as it might introduce data leakage. 

# Data Partitioning 
I will split my data into `training`, and `validation` subsets. 

`training` - will be used training algorithms with the data.

`validation` - will be used to test accuracy of models, choose the best model and tune it.



```python
# x and y variables 
y = df_model['isFraud']
x = df_model.drop(columns = ['isFraud'])
```

Due to the fact that our data is very large(~2million rows), I might have to reduce the data by reducing number of non-fraud transactions (there are only ~8000 fraud records, the rest is non-fraud).


```python
## convert index variable to dataframe
y_index = y.to_frame()

# non-fraud rows 
non_fraud_rows = y_index[y_index['isFraud'] == 0].index 

# percentage of removal 
percentage_remove = 0.50

# calculate number of exact number of rows that are going to be removed
rows = int(len(non_fraud_rows)*percentage_remove)

print(f"Number of rows that are going to be removed:{rows}")
```

    Number of rows that are going to be removed:1137678
    


```python
# random selection of non-fraud rows 
rows_remove = non_fraud_rows.to_series().sample(n = rows, random_state = 1042).index 

# apply on x and y 
x_reduced = x.drop(index = rows_remove)
y_reduced = y.drop(index = rows_remove)
```


```python
# info about data
x_reduced.info()
# 1145891 
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1145862 entries, 0 to 6362619
    Data columns (total 7 columns):
     #   Column          Non-Null Count    Dtype  
    ---  ------          --------------    -----  
     0   type            1145862 non-null  object 
     1   amount          1145862 non-null  float64
     2   oldbalanceOrg   1145862 non-null  float64
     3   newbalanceOrig  1145862 non-null  float64
     4   nameDest        1145862 non-null  object 
     5   oldbalanceDest  1145862 non-null  float64
     6   newbalanceDest  1145862 non-null  float64
    dtypes: float64(5), object(2)
    memory usage: 69.9+ MB
    


```python
# split data into training, and validation data
train_x, valid_x, train_y, valid_y = train_test_split(
    x_reduced, y_reduced, test_size = 0.4, random_state = 1042)
```


```python
# train dataset
print('Training dataset:')
train_x.info()
```

    Training dataset:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 687517 entries, 2122613 to 12901
    Data columns (total 7 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   type            687517 non-null  object 
     1   amount          687517 non-null  float64
     2   oldbalanceOrg   687517 non-null  float64
     3   newbalanceOrig  687517 non-null  float64
     4   nameDest        687517 non-null  object 
     5   oldbalanceDest  687517 non-null  float64
     6   newbalanceDest  687517 non-null  float64
    dtypes: float64(5), object(2)
    memory usage: 42.0+ MB
    


```python
# validation dataset
print('Validation dataset:')
valid_x.info()
```

    Validation dataset:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 458345 entries, 4544525 to 2692077
    Data columns (total 7 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   type            458345 non-null  object 
     1   amount          458345 non-null  float64
     2   oldbalanceOrg   458345 non-null  float64
     3   newbalanceOrig  458345 non-null  float64
     4   nameDest        458345 non-null  object 
     5   oldbalanceDest  458345 non-null  float64
     6   newbalanceDest  458345 non-null  float64
    dtypes: float64(5), object(2)
    memory usage: 28.0+ MB
    

Before I start cleaning each data subset, I want to make initial test of models performance with this state of data. 
At first, I will use logistic regression and single decision tree models without any tuning. 


```python
# list that will have all possible models 
models = []

# append models that are going to be used 
## logistic regression
models.append(('Logistic Regression', LogisticRegression(max_iter = 1000)))
## decision tree
models.append(('Single Decision Tree', DecisionTreeClassifier(random_state = 1042)))

# function to test performance of the model
def models_test(x, y, models, train_x, train_y, valid_x, valid_y):
  for name, model in models:
    # fit the model
    model.fit(train_x, train_y)
    
    # statistics for training data 
    print(f"Model {name} training Accuracy: {accuracy_score(train_y, model.predict(train_x)):.4f}")
    print(f"Model {name} training Precision: {precision_score(train_y, model.predict(train_x)):.4f}")
    print(f"Model {name} training Recall: {recall_score(train_y, model.predict(train_x)):.4f}")
    print(f"Model {name} training F1 Score: {f1_score(train_y, model.predict(train_x)):.4f}")
    ## ROC curve 
    ROC = model.predict_proba(train_x)[:, 1]
    print(f"Model {name} training ROC AUC: {roc_auc_score(train_y, ROC):.4f}")
    print()
    
    # classification summary for training data 
    classificationSummary(train_y,  model.predict(train_x))
    
    # statistics for validation data
    print(f"Model {name} validation Accuracy: {accuracy_score(valid_y, model.predict(valid_x)):.4f}")
    print(f"Model {name} validation Precision: {precision_score(valid_y, model.predict(valid_x)):.4f}")
    print(f"Model {name} validation Recall: {recall_score(valid_y, model.predict(valid_x)):.4f}")
    print(f"Model {name} validation F1 Score: {f1_score(valid_y, model.predict(valid_x)):.4f}")
    ## ROC curve 
    ROC = model.predict_proba(valid_x)[:, 1]
    print(f"Model {name} validation ROC AUC: {roc_auc_score(valid_y, ROC):.4f}")
    print()
        
    # classification summary for validation data 
    classificationSummary(valid_y, model.predict(valid_x))

    
    # Kfolds cross-validation 
    ## create cross-validation parameter (how many folds)
    cv = StratifiedKFold(n_splits = 5, 
                        shuffle = True, 
                        random_state = 1042)

    ## apply cross-validation statistics for data
    print(f"Model {name} Average cross-validation Accuracy: {cross_val_score(model,x, y, cv = 5, scoring = 'accuracy', n_jobs = -1).mean():.4f}")
    print(f"Model {name} Average cross-validation Precision: {cross_val_score(model,x, y, cv = 5, scoring = 'precision', n_jobs = -1).mean():.4f}")
    print(f"Model {name} Average cross-validation Recall: {cross_val_score(model,x, y, cv = 5, scoring = 'recall', n_jobs = -1).mean():.4f}")
    print(f"Model {name} Average cross-validation F1 Score: {cross_val_score(model,x, y, cv = 5, scoring = 'f1', n_jobs = -1).mean():.4f}")
    
    print('--------------------------------------------------------------------')
```


```python
# I am going to remove type object variable to test performance with only numeric variables at first 
## remove type variable 
x_copy = x_reduced.copy()

x_copy.drop(columns = ['type','nameDest'], axis = 1, inplace = True)

# split data into training, and validation data
train_x, valid_x, train_y, valid_y = train_test_split(
    x_copy, y_reduced, test_size = 0.4, random_state = 1042)

## test the model 
models_test(x_copy, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model Logistic Regression training Accuracy: 0.9963
    Model Logistic Regression training Precision: 0.8903
    Model Logistic Regression training Recall: 0.5441
    Model Logistic Regression training F1 Score: 0.6754
    Model Logistic Regression training ROC AUC: 0.9789
    
    Confusion Matrix (Accuracy 0.9963)
    
           Prediction
    Actual      0      1
         0 682283    329
         1   2236   2669
    Model Logistic Regression validation Accuracy: 0.9961
    Model Logistic Regression validation Precision: 0.8999
    Model Logistic Regression validation Recall: 0.5154
    Model Logistic Regression validation F1 Score: 0.6554
    Model Logistic Regression validation ROC AUC: 0.9785
    
    Confusion Matrix (Accuracy 0.9961)
    
           Prediction
    Actual      0      1
         0 454878    188
         1   1589   1690
    Model Logistic Regression Average cross-validation Accuracy: 0.9958
    Model Logistic Regression Average cross-validation Precision: 0.8594
    Model Logistic Regression Average cross-validation Recall: 0.5390
    Model Logistic Regression Average cross-validation F1 Score: 0.6554
    --------------------------------------------------------------------
    Model Single Decision Tree training Accuracy: 1.0000
    Model Single Decision Tree training Precision: 1.0000
    Model Single Decision Tree training Recall: 1.0000
    Model Single Decision Tree training F1 Score: 1.0000
    Model Single Decision Tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model Single Decision Tree validation Accuracy: 1.0000
    Model Single Decision Tree validation Precision: 0.9994
    Model Single Decision Tree validation Recall: 0.9991
    Model Single Decision Tree validation F1 Score: 0.9992
    Model Single Decision Tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455064      2
         1      3   3276
    Model Single Decision Tree Average cross-validation Accuracy: 1.0000
    Model Single Decision Tree Average cross-validation Precision: 0.9979
    Model Single Decision Tree Average cross-validation Recall: 0.9993
    Model Single Decision Tree Average cross-validation F1 Score: 0.9986
    --------------------------------------------------------------------
    

Without any tuning or critical data cleaning, decision tree already has a much better performance than logistic regression, even if decision tree is full-grown and clearly overfits to the training data. I am going to use decision trees models only if I wouldn't be able to improve the results of logistic regression. It is preffered to use logistic regression, as it easier to make regression interpretable and explainable in complex data. 

# Data Normalization / Scaling 


```python
# statistics about predictor variables 
x_copy.describe()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.145862e+06</td>
      <td>1.145862e+06</td>
      <td>1.145862e+06</td>
      <td>1.145862e+06</td>
      <td>1.145862e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.827244e+04</td>
      <td>2.291875e+06</td>
      <td>2.326684e+06</td>
      <td>9.072873e+05</td>
      <td>8.730499e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.586134e+05</td>
      <td>4.460921e+06</td>
      <td>4.512611e+06</td>
      <td>2.887001e+06</td>
      <td>2.881792e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.280702e+03</td>
      <td>5.365257e+04</td>
      <td>3.825763e+04</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.380047e+04</td>
      <td>2.364182e+05</td>
      <td>2.020774e+05</td>
      <td>0.000000e+00</td>
      <td>1.614095e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.187135e+05</td>
      <td>2.464495e+06</td>
      <td>2.600823e+06</td>
      <td>7.144960e+05</td>
      <td>6.216377e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+07</td>
      <td>5.958504e+07</td>
      <td>4.958504e+07</td>
      <td>3.114049e+08</td>
      <td>3.114929e+08</td>
    </tr>
  </tbody>
</table>
</div>



I am going to apply 2 types of scaling to numeric data: 
1) standard z normalization 

2) min-max normalization

I will choose normalization type that produces better results with decision trees. 

## Not normalized vs Z-normalized


```python
# library for pipelines
from sklearn.pipeline import Pipeline

# list that will have all possible models 
models = []

# pipelines 
tree_pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('decision_tree', DecisionTreeClassifier(random_state=1042))
])

logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('logistic_regression', LogisticRegression())
])

# append models that are going to be used 
## decision tree
models.append(('Single Decision Tree', DecisionTreeClassifier(random_state = 1042)))
models.append(('Single Decision Tree (standardized)', tree_pipeline))

## logistic 
models.append(('Logistic Regression (standardized)', logistic_pipeline))

```


```python
# test raw vs z-normalized 
models_test(x_copy, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model Single Decision Tree training Accuracy: 1.0000
    Model Single Decision Tree training Precision: 1.0000
    Model Single Decision Tree training Recall: 1.0000
    Model Single Decision Tree training F1 Score: 1.0000
    Model Single Decision Tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model Single Decision Tree validation Accuracy: 1.0000
    Model Single Decision Tree validation Precision: 0.9994
    Model Single Decision Tree validation Recall: 0.9991
    Model Single Decision Tree validation F1 Score: 0.9992
    Model Single Decision Tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455064      2
         1      3   3276
    Model Single Decision Tree Average cross-validation Accuracy: 1.0000
    Model Single Decision Tree Average cross-validation Precision: 0.9979
    Model Single Decision Tree Average cross-validation Recall: 0.9993
    Model Single Decision Tree Average cross-validation F1 Score: 0.9986
    --------------------------------------------------------------------
    Model Single Decision Tree (standardized) training Accuracy: 1.0000
    Model Single Decision Tree (standardized) training Precision: 1.0000
    Model Single Decision Tree (standardized) training Recall: 1.0000
    Model Single Decision Tree (standardized) training F1 Score: 1.0000
    Model Single Decision Tree (standardized) training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model Single Decision Tree (standardized) validation Accuracy: 1.0000
    Model Single Decision Tree (standardized) validation Precision: 0.9988
    Model Single Decision Tree (standardized) validation Recall: 0.9988
    Model Single Decision Tree (standardized) validation F1 Score: 0.9988
    Model Single Decision Tree (standardized) validation ROC AUC: 0.9994
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455062      4
         1      4   3275
    Model Single Decision Tree (standardized) Average cross-validation Accuracy: 1.0000
    Model Single Decision Tree (standardized) Average cross-validation Precision: 0.9981
    Model Single Decision Tree (standardized) Average cross-validation Recall: 0.9988
    Model Single Decision Tree (standardized) Average cross-validation F1 Score: 0.9984
    --------------------------------------------------------------------
    Model Logistic Regression (standardized) training Accuracy: 0.9959
    Model Logistic Regression (standardized) training Precision: 0.8883
    Model Logistic Regression (standardized) training Recall: 0.4834
    Model Logistic Regression (standardized) training F1 Score: 0.6261
    Model Logistic Regression (standardized) training ROC AUC: 0.9720
    
    Confusion Matrix (Accuracy 0.9959)
    
           Prediction
    Actual      0      1
         0 682314    298
         1   2534   2371
    Model Logistic Regression (standardized) validation Accuracy: 0.9958
    Model Logistic Regression (standardized) validation Precision: 0.8957
    Model Logistic Regression (standardized) validation Recall: 0.4608
    Model Logistic Regression (standardized) validation F1 Score: 0.6085
    Model Logistic Regression (standardized) validation ROC AUC: 0.9713
    
    Confusion Matrix (Accuracy 0.9958)
    
           Prediction
    Actual      0      1
         0 454890    176
         1   1768   1511
    Model Logistic Regression (standardized) Average cross-validation Accuracy: 0.9958
    Model Logistic Regression (standardized) Average cross-validation Precision: 0.8811
    Model Logistic Regression (standardized) Average cross-validation Recall: 0.4875
    Model Logistic Regression (standardized) Average cross-validation F1 Score: 0.6267
    --------------------------------------------------------------------
    

Decision tree has actually worse results when used with standardization. That means it is better to use decision tree without the standardization, unless if min-max standardization will show better results. Logistic regression has still less performance on validation data and recall rate worsened after the standardization too.

Accuracy metric is also not useful to me in this data, because it converges to 1 indifferently to any data transformation. I will focus on other three (Precision, Recall, F1 score) metrics in validation and cross-validation. 

## Z-Normalized vs Min-Max Normalized 



```python
# list that will have all possible models 
models = []

# pipelines 
tree_pipeline = Pipeline([
    ('decision_tree', DecisionTreeClassifier(random_state=1042))
])

tree_pipeline_minmax = Pipeline([
    ('scaler', MinMaxScaler() ), 
    ('decision_tree', DecisionTreeClassifier(random_state = 1042))
])

logistic_pipeline_minmax = Pipeline([
    ('scaler', MinMaxScaler()),  
    ('logistic_regression', LogisticRegression())
])

# append models that are going to be used 
## decision tree
models.append(('Single Decision Tree', tree_pipeline))
models.append(('Single Decision Tree (min-max normalization)', tree_pipeline_minmax))

## logistic 
models.append(('Logistic Regression (min-max normalization)', logistic_pipeline_minmax))

```


```python
# test normalized vs min-max normalized 
models_test(x_copy, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model Single Decision Tree training Accuracy: 1.0000
    Model Single Decision Tree training Precision: 1.0000
    Model Single Decision Tree training Recall: 1.0000
    Model Single Decision Tree training F1 Score: 1.0000
    Model Single Decision Tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model Single Decision Tree validation Accuracy: 1.0000
    Model Single Decision Tree validation Precision: 0.9994
    Model Single Decision Tree validation Recall: 0.9991
    Model Single Decision Tree validation F1 Score: 0.9992
    Model Single Decision Tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455064      2
         1      3   3276
    Model Single Decision Tree Average cross-validation Accuracy: 1.0000
    Model Single Decision Tree Average cross-validation Precision: 0.9979
    Model Single Decision Tree Average cross-validation Recall: 0.9993
    Model Single Decision Tree Average cross-validation F1 Score: 0.9986
    --------------------------------------------------------------------
    Model Single Decision Tree (min-max normalization) training Accuracy: 1.0000
    Model Single Decision Tree (min-max normalization) training Precision: 1.0000
    Model Single Decision Tree (min-max normalization) training Recall: 0.9998
    Model Single Decision Tree (min-max normalization) training F1 Score: 0.9999
    Model Single Decision Tree (min-max normalization) training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      1   4904
    Model Single Decision Tree (min-max normalization) validation Accuracy: 0.9999
    Model Single Decision Tree (min-max normalization) validation Precision: 0.9906
    Model Single Decision Tree (min-max normalization) validation Recall: 0.9915
    Model Single Decision Tree (min-max normalization) validation F1 Score: 0.9910
    Model Single Decision Tree (min-max normalization) validation ROC AUC: 0.9957
    
    Confusion Matrix (Accuracy 0.9999)
    
           Prediction
    Actual      0      1
         0 455035     31
         1     28   3251
    Model Single Decision Tree (min-max normalization) Average cross-validation Accuracy: 0.9999
    Model Single Decision Tree (min-max normalization) Average cross-validation Precision: 0.9927
    Model Single Decision Tree (min-max normalization) Average cross-validation Recall: 0.9917
    Model Single Decision Tree (min-max normalization) Average cross-validation F1 Score: 0.9922
    --------------------------------------------------------------------
    Model Logistic Regression (min-max normalization) training Accuracy: 0.9950
    Model Logistic Regression (min-max normalization) training Precision: 0.9867
    Model Logistic Regression (min-max normalization) training Recall: 0.3028
    Model Logistic Regression (min-max normalization) training F1 Score: 0.4633
    Model Logistic Regression (min-max normalization) training ROC AUC: 0.9557
    
    Confusion Matrix (Accuracy 0.9950)
    
           Prediction
    Actual      0      1
         0 682592     20
         1   3420   1485
    Model Logistic Regression (min-max normalization) validation Accuracy: 0.9948
    Model Logistic Regression (min-max normalization) validation Precision: 0.9841
    Model Logistic Regression (min-max normalization) validation Recall: 0.2824
    Model Logistic Regression (min-max normalization) validation F1 Score: 0.4389
    Model Logistic Regression (min-max normalization) validation ROC AUC: 0.9566
    
    Confusion Matrix (Accuracy 0.9948)
    
           Prediction
    Actual      0      1
         0 455051     15
         1   2353    926
    Model Logistic Regression (min-max normalization) Average cross-validation Accuracy: 0.9953
    Model Logistic Regression (min-max normalization) Average cross-validation Precision: 0.9794
    Model Logistic Regression (min-max normalization) Average cross-validation Recall: 0.3556
    Model Logistic Regression (min-max normalization) Average cross-validation F1 Score: 0.5216
    --------------------------------------------------------------------
    

Min-max normalization is clearly worse than z-normalization and no normalization for this data for both models. The possible reason is that numerical variables of this data are very right-skewed and have big ranges, z normalization is better for columns with different ranges. 


Based on the results, no normalization is better both for logistic regression and decision tree, so I am not going to use any type of normalization for deicision tree, but it is recommended to use z-normalization for logistic regression. 

# Categorical variables into Dummy variables
I am going to convert categorical variable - `Type` into dummy variable in x_copy. If this transformation improves predictive of power of models, I will transform `Type` in the main x_reduced variable. 


```python
# make a copy of x_reduced variable again without dropping type column
x_copy = x_reduced.copy()

```


```python
# information about x_copy
x_copy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1145862 entries, 0 to 6362619
    Data columns (total 7 columns):
     #   Column          Non-Null Count    Dtype  
    ---  ------          --------------    -----  
     0   type            1145862 non-null  object 
     1   amount          1145862 non-null  float64
     2   oldbalanceOrg   1145862 non-null  float64
     3   newbalanceOrig  1145862 non-null  float64
     4   nameDest        1145862 non-null  object 
     5   oldbalanceDest  1145862 non-null  float64
     6   newbalanceDest  1145862 non-null  float64
    dtypes: float64(5), object(2)
    memory usage: 69.9+ MB
    

From the data exploration part, I learned that Merchants in `nameDest` are always not Fraud, I am going to transform this variable and convert this columns to use it efficiently. I am going to rename every row in this column to either `Consumer` or `Merchant`, and then I will convert this column to dummy variable. 


```python
# convert rows in the column 
x_copy['nameDest'] = np.where(x_copy['nameDest'].str.startswith('C'), 'Consumer', 
                             np.where(x_copy['nameDest'].str.startswith('M'), 'Merchant', x_copy['nameDest']))

# check 
x_copy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PAYMENT</td>
      <td>9839.64</td>
      <td>170136.0</td>
      <td>160296.36</td>
      <td>Merchant</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PAYMENT</td>
      <td>1864.28</td>
      <td>21249.0</td>
      <td>19384.72</td>
      <td>Merchant</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRANSFER</td>
      <td>181.00</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>Consumer</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CASH_OUT</td>
      <td>181.00</td>
      <td>181.0</td>
      <td>0.00</td>
      <td>Consumer</td>
      <td>21182.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAYMENT</td>
      <td>11668.14</td>
      <td>41554.0</td>
      <td>29885.86</td>
      <td>Merchant</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert `type` to dummy
x_copy = pd.get_dummies(x_copy, drop_first = True)
```


```python
# partition the data 
train_x, valid_x, train_y, valid_y = train_test_split(
    x_copy, y_reduced, test_size = 0.40, random_state = 1042)
```


```python
# check
train_x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 687517 entries, 2122613 to 12901
    Data columns (total 10 columns):
     #   Column             Non-Null Count   Dtype  
    ---  ------             --------------   -----  
     0   amount             687517 non-null  float64
     1   oldbalanceOrg      687517 non-null  float64
     2   newbalanceOrig     687517 non-null  float64
     3   oldbalanceDest     687517 non-null  float64
     4   newbalanceDest     687517 non-null  float64
     5   type_CASH_OUT      687517 non-null  uint8  
     6   type_DEBIT         687517 non-null  uint8  
     7   type_PAYMENT       687517 non-null  uint8  
     8   type_TRANSFER      687517 non-null  uint8  
     9   nameDest_Merchant  687517 non-null  uint8  
    dtypes: float64(5), uint8(5)
    memory usage: 34.8 MB
    


```python
# data rows 
train_x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>type_CASH_OUT</th>
      <th>type_DEBIT</th>
      <th>type_PAYMENT</th>
      <th>type_TRANSFER</th>
      <th>nameDest_Merchant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2122613</th>
      <td>180821.38</td>
      <td>3401538.73</td>
      <td>3582360.10</td>
      <td>1857943.23</td>
      <td>1677121.85</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4486532</th>
      <td>37622.60</td>
      <td>1598811.11</td>
      <td>1636433.71</td>
      <td>95064.21</td>
      <td>57441.61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4305264</th>
      <td>85323.89</td>
      <td>11474070.89</td>
      <td>11559394.78</td>
      <td>172909.95</td>
      <td>87586.06</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2454200</th>
      <td>426149.05</td>
      <td>750013.95</td>
      <td>1176163.00</td>
      <td>2962209.25</td>
      <td>2536060.19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1039716</th>
      <td>5111.63</td>
      <td>11466.00</td>
      <td>6354.37</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check 
valid_x.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 458345 entries, 4544525 to 2692077
    Data columns (total 10 columns):
     #   Column             Non-Null Count   Dtype  
    ---  ------             --------------   -----  
     0   amount             458345 non-null  float64
     1   oldbalanceOrg      458345 non-null  float64
     2   newbalanceOrig     458345 non-null  float64
     3   oldbalanceDest     458345 non-null  float64
     4   newbalanceDest     458345 non-null  float64
     5   type_CASH_OUT      458345 non-null  uint8  
     6   type_DEBIT         458345 non-null  uint8  
     7   type_PAYMENT       458345 non-null  uint8  
     8   type_TRANSFER      458345 non-null  uint8  
     9   nameDest_Merchant  458345 non-null  uint8  
    dtypes: float64(5), uint8(5)
    memory usage: 23.2 MB
    


```python
# data rows 
valid_x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>type_CASH_OUT</th>
      <th>type_DEBIT</th>
      <th>type_PAYMENT</th>
      <th>type_TRANSFER</th>
      <th>nameDest_Merchant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4544525</th>
      <td>30185.02</td>
      <td>167656.00</td>
      <td>137470.98</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2212018</th>
      <td>531346.79</td>
      <td>3111959.43</td>
      <td>3643306.22</td>
      <td>4751930.73</td>
      <td>3910486.42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4727737</th>
      <td>3324.38</td>
      <td>58082.87</td>
      <td>54758.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5919277</th>
      <td>319390.80</td>
      <td>3372715.40</td>
      <td>3692106.21</td>
      <td>615357.02</td>
      <td>295966.21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3584576</th>
      <td>14009.50</td>
      <td>739835.00</td>
      <td>725825.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# list that will have all possible models 
models = []

# pipelines 
tree_pipeline = Pipeline([
    ('decision_tree', DecisionTreeClassifier(random_state=1042))
])

logistic_pipeline_z = Pipeline([
    ('scaler', StandardScaler()),  
    ('logistic_regression', LogisticRegression(random_state=1042))
])
    
    
## decision tree
models.append(('Single Decision Tree', tree_pipeline))
    
## logistic 
models.append(('Logistic Regression (standardized)', logistic_pipeline_z))

```


```python
# test decision tree with dummy variables  
models_test(x_copy, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model Single Decision Tree training Accuracy: 1.0000
    Model Single Decision Tree training Precision: 1.0000
    Model Single Decision Tree training Recall: 1.0000
    Model Single Decision Tree training F1 Score: 1.0000
    Model Single Decision Tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model Single Decision Tree validation Accuracy: 1.0000
    Model Single Decision Tree validation Precision: 0.9991
    Model Single Decision Tree validation Recall: 0.9991
    Model Single Decision Tree validation F1 Score: 0.9991
    Model Single Decision Tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455063      3
         1      3   3276
    Model Single Decision Tree Average cross-validation Accuracy: 1.0000
    Model Single Decision Tree Average cross-validation Precision: 0.9989
    Model Single Decision Tree Average cross-validation Recall: 0.9993
    Model Single Decision Tree Average cross-validation F1 Score: 0.9991
    --------------------------------------------------------------------
    Model Logistic Regression (standardized) training Accuracy: 0.9960
    Model Logistic Regression (standardized) training Precision: 0.8798
    Model Logistic Regression (standardized) training Recall: 0.5015
    Model Logistic Regression (standardized) training F1 Score: 0.6389
    Model Logistic Regression (standardized) training ROC AUC: 0.9905
    
    Confusion Matrix (Accuracy 0.9960)
    
           Prediction
    Actual      0      1
         0 682276    336
         1   2445   2460
    Model Logistic Regression (standardized) validation Accuracy: 0.9958
    Model Logistic Regression (standardized) validation Precision: 0.8820
    Model Logistic Regression (standardized) validation Recall: 0.4785
    Model Logistic Regression (standardized) validation F1 Score: 0.6204
    Model Logistic Regression (standardized) validation ROC AUC: 0.9900
    
    Confusion Matrix (Accuracy 0.9958)
    
           Prediction
    Actual      0      1
         0 454856    210
         1   1710   1569
    Model Logistic Regression (standardized) Average cross-validation Accuracy: 0.9959
    Model Logistic Regression (standardized) Average cross-validation Precision: 0.8809
    Model Logistic Regression (standardized) Average cross-validation Recall: 0.5012
    Model Logistic Regression (standardized) Average cross-validation F1 Score: 0.6384
    --------------------------------------------------------------------
    

Dummy variables improved validation metrics in decision tree model. There is an increase in all metrics in cross-validation metrics, which is a good sign. However, there was no improvement in logistic regression model. Oppositely, the results became worse. As there is an improvement in results of decision tree, I will copy all of the results to main variable - x_reduced. Decision tree is better for such data, that's why I will use and tune decision tree only from now on. 


```python
# copy changed in x_copy
x_reduced = x_copy.copy()
```


```python
# check 
x_reduced.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1145862 entries, 0 to 6362619
    Data columns (total 10 columns):
     #   Column             Non-Null Count    Dtype  
    ---  ------             --------------    -----  
     0   amount             1145862 non-null  float64
     1   oldbalanceOrg      1145862 non-null  float64
     2   newbalanceOrig     1145862 non-null  float64
     3   oldbalanceDest     1145862 non-null  float64
     4   newbalanceDest     1145862 non-null  float64
     5   type_CASH_OUT      1145862 non-null  uint8  
     6   type_DEBIT         1145862 non-null  uint8  
     7   type_PAYMENT       1145862 non-null  uint8  
     8   type_TRANSFER      1145862 non-null  uint8  
     9   nameDest_Merchant  1145862 non-null  uint8  
    dtypes: float64(5), uint8(5)
    memory usage: 57.9 MB
    

# Target class over-sampling
Next step to try and improve the model is to oversample the number of fraud transactions in the data. Currently, there are only ~8000 rows in the data that correspond to fraud transactions. This might have been worsening the prediction power of models. 

I will use 2 methods of oversampling: 
1) SMOTE: Synthetic Minority Oversampling Technique

2) ADASYN: Adaptive Synthetic Sampling Approach

Best Oversampling method will be chosen based on validation and cross-validation metrics.

( There are much better combined-hybrid oversampling techniques that give better performance, however they are computationally expensive and can't be done on millions of rows without additional resources which I don't have)

## SMOTE 


```python
# library for oversampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# list that will have all possible models 
models = []

# pipelines 
tree_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state = 1042)), 
    ('decision_tree', DecisionTreeClassifier(random_state=1042))
])

# append models that are going to be used 
## decision tree
models.append(('Single Decision Tree', tree_pipeline))
```


```python
# partition the data 
train_x, valid_x, train_y, valid_y = train_test_split(
    x_reduced, y_reduced, test_size = 0.40, random_state = 1042)
```

Previous result: 
![image-2.png](attachment:image-2.png)


```python
# test decision tree (SMOTE method)
models_test(x_reduced, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model Single Decision Tree training Accuracy: 1.0000
    Model Single Decision Tree training Precision: 1.0000
    Model Single Decision Tree training Recall: 1.0000
    Model Single Decision Tree training F1 Score: 1.0000
    Model Single Decision Tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model Single Decision Tree validation Accuracy: 1.0000
    Model Single Decision Tree validation Precision: 0.9997
    Model Single Decision Tree validation Recall: 0.9991
    Model Single Decision Tree validation F1 Score: 0.9994
    Model Single Decision Tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455065      1
         1      3   3276
    Model Single Decision Tree Average cross-validation Accuracy: 1.0000
    Model Single Decision Tree Average cross-validation Precision: 0.9985
    Model Single Decision Tree Average cross-validation Recall: 0.9995
    Model Single Decision Tree Average cross-validation F1 Score: 0.9990
    --------------------------------------------------------------------
    

## ADASYN


```python
# list that will have all possible models 
models = []

# pipelines 
tree_pipeline = ImbPipeline([
    ('adasyn', ADASYN(random_state = 1042)), 
    ('decision_tree', DecisionTreeClassifier(random_state=1042))
])

# append models that are going to be used 
## decision tree
models.append(('single Decision Tree', tree_pipeline))

```


```python
# test decision tree (ADASYN method)   
models_test(x_reduced, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model single Decision Tree training Accuracy: 1.0000
    Model single Decision Tree training Precision: 1.0000
    Model single Decision Tree training Recall: 1.0000
    Model single Decision Tree training F1 Score: 1.0000
    Model single Decision Tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model single Decision Tree validation Accuracy: 1.0000
    Model single Decision Tree validation Precision: 0.9997
    Model single Decision Tree validation Recall: 0.9991
    Model single Decision Tree validation F1 Score: 0.9994
    Model single Decision Tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455065      1
         1      3   3276
    Model single Decision Tree Average cross-validation Accuracy: 1.0000
    Model single Decision Tree Average cross-validation Precision: 0.9980
    Model single Decision Tree Average cross-validation Recall: 0.9995
    Model single Decision Tree Average cross-validation F1 Score: 0.9988
    --------------------------------------------------------------------
    

SMOTE has better improvement of the model predictive power. It doesn't have any improvement in validation metrics, but it clearly has improvement in cross-validation metrics which is more important. Now I will try to make model even better by tuning it and improving interpretability. 

# Model Tuning 

## Best Parameters


```python
tree_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state = 1042)), 
    ('decision_tree', DecisionTreeClassifier(random_state=1042))
])

# feet the tree again
tree_pipeline.fit(train_x, train_y)

# current depth of the tree 
tree_depth = tree_pipeline.named_steps['decision_tree'].get_depth()

# print
print(tree_depth)
```

    4
    


```python
plt.figure(figsize=(20,12))
plot_tree(tree_pipeline.named_steps['decision_tree'], 
          filled=True, 
         feature_names = train_x.columns.tolist(), 
         class_names = list(map(str, train_y.unique().tolist())))
          
plt.show()
```


    
![png](output_117_0.png)
    


Surprisingly, "full-grown" tree has only 4 as a maximum depth, which is not large. It is also very explainable and steps are relatively logical. However, I will try to improve the decision tree even more. I will try to apply thresholds to decrease FP and FN rates. 




```python
# list that will have all possible models 
models = []


## decision tree pipeline
tree_pipeline_best =  ImbPipeline([
    ('smote', SMOTE(random_state = 1042)),
    ('decision_tree', DecisionTreeClassifier(random_state=1042))
])

# append models that are going to be used 
models.append(('best decision tree', tree_pipeline_best))

```


```python
# test decision tree (SMOTE method)   
## just to have recent results near
models_test(x_reduced, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model best decision tree training Accuracy: 1.0000
    Model best decision tree training Precision: 1.0000
    Model best decision tree training Recall: 1.0000
    Model best decision tree training F1 Score: 1.0000
    Model best decision tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682612      0
         1      0   4905
    Model best decision tree validation Accuracy: 1.0000
    Model best decision tree validation Precision: 0.9997
    Model best decision tree validation Recall: 0.9991
    Model best decision tree validation F1 Score: 0.9994
    Model best decision tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455065      1
         1      3   3276
    Model best decision tree Average cross-validation Accuracy: 1.0000
    Model best decision tree Average cross-validation Precision: 0.9985
    Model best decision tree Average cross-validation Recall: 0.9995
    Model best decision tree Average cross-validation F1 Score: 0.9990
    --------------------------------------------------------------------
    


```python
# fit
tree_pipeline_best.fit(train_x, train_y)


# thresholds
thresholds = np.linspace(0, 1, num=100)

# true labels
true_labels = []

# predict probabilities 
predicted_probabilities = tree_pipeline_best.predict_proba(valid_x)[:, 1]

# lists with rates 
fn_rates = []
fp_rates = []
fn_fp_list = []

# rates for each threshold (for loop)
for threshold in thresholds: 
    predicted_labels = (predicted_probabilities > threshold).astype(int)
    
    # calculate metrics 
    tn, fp, fn, tp = confusion_matrix(valid_y, predicted_labels).ravel()
    
    
    # append 
    fn_rate = fn / (fn + tp)
    fp_rate = fp / (fp + tn)
    fn_rates.append(fn_rate)
    fp_rates.append(fp_rate)
    
    # append thresholds and results 
    fn_fp_list.append((threshold, fp, fn))

# print 
for item in fn_fp_list:
    print(f'Threshold: {item[0]:.2f}, FP: {item[1]}, FN: {item[2]}')
```

    Threshold: 0.00, FP: 1, FN: 3
    Threshold: 0.01, FP: 1, FN: 3
    Threshold: 0.02, FP: 1, FN: 3
    Threshold: 0.03, FP: 1, FN: 3
    Threshold: 0.04, FP: 1, FN: 3
    Threshold: 0.05, FP: 1, FN: 3
    Threshold: 0.06, FP: 1, FN: 3
    Threshold: 0.07, FP: 1, FN: 3
    Threshold: 0.08, FP: 1, FN: 3
    Threshold: 0.09, FP: 1, FN: 3
    Threshold: 0.10, FP: 1, FN: 3
    Threshold: 0.11, FP: 1, FN: 3
    Threshold: 0.12, FP: 1, FN: 3
    Threshold: 0.13, FP: 1, FN: 3
    Threshold: 0.14, FP: 1, FN: 3
    Threshold: 0.15, FP: 1, FN: 3
    Threshold: 0.16, FP: 1, FN: 3
    Threshold: 0.17, FP: 1, FN: 3
    Threshold: 0.18, FP: 1, FN: 3
    Threshold: 0.19, FP: 1, FN: 3
    Threshold: 0.20, FP: 1, FN: 3
    Threshold: 0.21, FP: 1, FN: 3
    Threshold: 0.22, FP: 1, FN: 3
    Threshold: 0.23, FP: 1, FN: 3
    Threshold: 0.24, FP: 1, FN: 3
    Threshold: 0.25, FP: 1, FN: 3
    Threshold: 0.26, FP: 1, FN: 3
    Threshold: 0.27, FP: 1, FN: 3
    Threshold: 0.28, FP: 1, FN: 3
    Threshold: 0.29, FP: 1, FN: 3
    Threshold: 0.30, FP: 1, FN: 3
    Threshold: 0.31, FP: 1, FN: 3
    Threshold: 0.32, FP: 1, FN: 3
    Threshold: 0.33, FP: 1, FN: 3
    Threshold: 0.34, FP: 1, FN: 3
    Threshold: 0.35, FP: 1, FN: 3
    Threshold: 0.36, FP: 1, FN: 3
    Threshold: 0.37, FP: 1, FN: 3
    Threshold: 0.38, FP: 1, FN: 3
    Threshold: 0.39, FP: 1, FN: 3
    Threshold: 0.40, FP: 1, FN: 3
    Threshold: 0.41, FP: 1, FN: 3
    Threshold: 0.42, FP: 1, FN: 3
    Threshold: 0.43, FP: 1, FN: 3
    Threshold: 0.44, FP: 1, FN: 3
    Threshold: 0.45, FP: 1, FN: 3
    Threshold: 0.46, FP: 1, FN: 3
    Threshold: 0.47, FP: 1, FN: 3
    Threshold: 0.48, FP: 1, FN: 3
    Threshold: 0.49, FP: 1, FN: 3
    Threshold: 0.51, FP: 1, FN: 3
    Threshold: 0.52, FP: 1, FN: 3
    Threshold: 0.53, FP: 1, FN: 3
    Threshold: 0.54, FP: 1, FN: 3
    Threshold: 0.55, FP: 1, FN: 3
    Threshold: 0.56, FP: 1, FN: 3
    Threshold: 0.57, FP: 1, FN: 3
    Threshold: 0.58, FP: 1, FN: 3
    Threshold: 0.59, FP: 1, FN: 3
    Threshold: 0.60, FP: 1, FN: 3
    Threshold: 0.61, FP: 1, FN: 3
    Threshold: 0.62, FP: 1, FN: 3
    Threshold: 0.63, FP: 1, FN: 3
    Threshold: 0.64, FP: 1, FN: 3
    Threshold: 0.65, FP: 1, FN: 3
    Threshold: 0.66, FP: 1, FN: 3
    Threshold: 0.67, FP: 1, FN: 3
    Threshold: 0.68, FP: 1, FN: 3
    Threshold: 0.69, FP: 1, FN: 3
    Threshold: 0.70, FP: 1, FN: 3
    Threshold: 0.71, FP: 1, FN: 3
    Threshold: 0.72, FP: 1, FN: 3
    Threshold: 0.73, FP: 1, FN: 3
    Threshold: 0.74, FP: 1, FN: 3
    Threshold: 0.75, FP: 1, FN: 3
    Threshold: 0.76, FP: 1, FN: 3
    Threshold: 0.77, FP: 1, FN: 3
    Threshold: 0.78, FP: 1, FN: 3
    Threshold: 0.79, FP: 1, FN: 3
    Threshold: 0.80, FP: 1, FN: 3
    Threshold: 0.81, FP: 1, FN: 3
    Threshold: 0.82, FP: 1, FN: 3
    Threshold: 0.83, FP: 1, FN: 3
    Threshold: 0.84, FP: 1, FN: 3
    Threshold: 0.85, FP: 1, FN: 3
    Threshold: 0.86, FP: 1, FN: 3
    Threshold: 0.87, FP: 1, FN: 3
    Threshold: 0.88, FP: 1, FN: 3
    Threshold: 0.89, FP: 1, FN: 3
    Threshold: 0.90, FP: 1, FN: 3
    Threshold: 0.91, FP: 1, FN: 3
    Threshold: 0.92, FP: 1, FN: 3
    Threshold: 0.93, FP: 1, FN: 3
    Threshold: 0.94, FP: 1, FN: 3
    Threshold: 0.95, FP: 1, FN: 3
    Threshold: 0.96, FP: 1, FN: 3
    Threshold: 0.97, FP: 1, FN: 3
    Threshold: 0.98, FP: 1, FN: 3
    Threshold: 0.99, FP: 1, FN: 3
    Threshold: 1.00, FP: 0, FN: 3279
    

Unexpected result as threshold can't find an optimal value where the FP and FN rates will be less. However, the decision tree has a very good performance. Last thing that can be tried is RandomSearch of hyperparameters. Random search is better than Grid-Search as Grid-Search is computationally intensive with big data.  


```python
# random search parameters 
parameters = {
    'decision_tree__max_depth': [ 3, 4, 5],
    'decision_tree__max_features': ['auto', 'sqrt', 'log2', None],
    'decision_tree__criterion': ['gini', 'entropy', 'log_reduction'],
    'decision_tree__min_samples_leaf': [1, 2, 3, 4, 5, 6],
    'decision_tree__min_samples_split': np.arange(2, 20, step=2)
}

# launch randomsearch 
random_search = RandomizedSearchCV(
            tree_pipeline_best, 
            param_distributions=parameters, 
            n_iter=11, 
            cv=5, 
            verbose=1, 
            random_state=1042, 
            n_jobs=-1)

# fit the random search 
random_search.fit(train_x, train_y)
```

    Fitting 5 folds for each of 11 candidates, totalling 55 fits
    

    C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py:547: FitFailedWarning: 
    25 fits failed out of a total of 55.
    The score on these train-test partitions for these parameters will be set to nan.
    If these failures are not expected, you can try to debug them by setting error_score='raise'.
    
    Below are more details about the failures:
    --------------------------------------------------------------------------------
    6 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\imblearn\pipeline.py", line 326, in fit
        self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1467, in wrapper
        estimator._validate_params()
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 666, in _validate_params
        validate_parameter_constraints(
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
        raise InvalidParameterError(
    sklearn.utils._param_validation.InvalidParameterError: The 'criterion' parameter of DecisionTreeClassifier must be a str among {'entropy', 'gini', 'log_loss'}. Got 'log_reduction' instead.
    
    --------------------------------------------------------------------------------
    2 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\imblearn\pipeline.py", line 326, in fit
        self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1467, in wrapper
        estimator._validate_params()
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 666, in _validate_params
        validate_parameter_constraints(
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
        raise InvalidParameterError(
    sklearn.utils._param_validation.InvalidParameterError: The 'criterion' parameter of DecisionTreeClassifier must be a str among {'gini', 'entropy', 'log_loss'}. Got 'log_reduction' instead.
    
    --------------------------------------------------------------------------------
    4 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\imblearn\pipeline.py", line 326, in fit
        self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1467, in wrapper
        estimator._validate_params()
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 666, in _validate_params
        validate_parameter_constraints(
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
        raise InvalidParameterError(
    sklearn.utils._param_validation.InvalidParameterError: The 'criterion' parameter of DecisionTreeClassifier must be a str among {'log_loss', 'gini', 'entropy'}. Got 'log_reduction' instead.
    
    --------------------------------------------------------------------------------
    2 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\imblearn\pipeline.py", line 326, in fit
        self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1467, in wrapper
        estimator._validate_params()
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 666, in _validate_params
        validate_parameter_constraints(
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
        raise InvalidParameterError(
    sklearn.utils._param_validation.InvalidParameterError: The 'criterion' parameter of DecisionTreeClassifier must be a str among {'entropy', 'log_loss', 'gini'}. Got 'log_reduction' instead.
    
    --------------------------------------------------------------------------------
    1 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\imblearn\pipeline.py", line 326, in fit
        self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1467, in wrapper
        estimator._validate_params()
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 666, in _validate_params
        validate_parameter_constraints(
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
        raise InvalidParameterError(
    sklearn.utils._param_validation.InvalidParameterError: The 'criterion' parameter of DecisionTreeClassifier must be a str among {'gini', 'log_loss', 'entropy'}. Got 'log_reduction' instead.
    
    --------------------------------------------------------------------------------
    6 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\imblearn\pipeline.py", line 326, in fit
        self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1467, in wrapper
        estimator._validate_params()
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 666, in _validate_params
        validate_parameter_constraints(
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
        raise InvalidParameterError(
    sklearn.utils._param_validation.InvalidParameterError: The 'max_features' parameter of DecisionTreeClassifier must be an int in the range [1, inf), a float in the range (0.0, 1.0], a str among {'log2', 'sqrt'} or None. Got 'auto' instead.
    
    --------------------------------------------------------------------------------
    4 fits failed with the following error:
    Traceback (most recent call last):
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_validation.py", line 895, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1474, in wrapper
        return fit_method(estimator, *args, **kwargs)
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\imblearn\pipeline.py", line 326, in fit
        self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 1467, in wrapper
        estimator._validate_params()
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\base.py", line 666, in _validate_params
        validate_parameter_constraints(
      File "C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
        raise InvalidParameterError(
    sklearn.utils._param_validation.InvalidParameterError: The 'max_features' parameter of DecisionTreeClassifier must be an int in the range [1, inf), a float in the range (0.0, 1.0], a str among {'sqrt', 'log2'} or None. Got 'auto' instead.
    
      warnings.warn(some_fits_failed_message, FitFailedWarning)
    C:\Users\Beibarys Nyussupov\anaconda3\lib\site-packages\sklearn\model_selection\_search.py:1051: UserWarning: One or more of the test scores are non-finite: [       nan        nan        nan 0.99999418 0.99147219 0.99987637
     0.99997527        nan 0.99987637        nan 0.99999418]
      warnings.warn(
    




<style>#sk-container-id-13 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-13 {
  color: var(--sklearn-color-text);
}

#sk-container-id-13 pre {
  padding: 0;
}

#sk-container-id-13 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-13 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-13 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-13 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-13 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-13 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-13 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-13 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-13 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-13 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-13 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-13 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-13 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-13 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-13 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-13 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-13 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-13 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-13 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-13 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-13 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-13 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-13 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-13 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-13 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-13 div.sk-label label.sk-toggleable__label,
#sk-container-id-13 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-13 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-13 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-13 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-13 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-13 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-13 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-13 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-13 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-13 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-13 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-13 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-13 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-13" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=5,
                   estimator=Pipeline(steps=[(&#x27;smote&#x27;,
                                              SMOTE(random_state=1042)),
                                             (&#x27;decision_tree&#x27;,
                                              DecisionTreeClassifier(random_state=1042))]),
                   n_iter=11, n_jobs=-1,
                   param_distributions={&#x27;decision_tree__criterion&#x27;: [&#x27;gini&#x27;,
                                                                     &#x27;entropy&#x27;,
                                                                     &#x27;log_reduction&#x27;],
                                        &#x27;decision_tree__max_depth&#x27;: [3, 4, 5],
                                        &#x27;decision_tree__max_features&#x27;: [&#x27;auto&#x27;,
                                                                        &#x27;sqrt&#x27;,
                                                                        &#x27;log2&#x27;,
                                                                        None],
                                        &#x27;decision_tree__min_samples_leaf&#x27;: [1,
                                                                            2,
                                                                            3,
                                                                            4,
                                                                            5,
                                                                            6],
                                        &#x27;decision_tree__min_samples_split&#x27;: array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])},
                   random_state=1042, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-49" type="checkbox" ><label for="sk-estimator-id-49" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomizedSearchCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.RandomizedSearchCV.html">?<span>Documentation for RandomizedSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomizedSearchCV(cv=5,
                   estimator=Pipeline(steps=[(&#x27;smote&#x27;,
                                              SMOTE(random_state=1042)),
                                             (&#x27;decision_tree&#x27;,
                                              DecisionTreeClassifier(random_state=1042))]),
                   n_iter=11, n_jobs=-1,
                   param_distributions={&#x27;decision_tree__criterion&#x27;: [&#x27;gini&#x27;,
                                                                     &#x27;entropy&#x27;,
                                                                     &#x27;log_reduction&#x27;],
                                        &#x27;decision_tree__max_depth&#x27;: [3, 4, 5],
                                        &#x27;decision_tree__max_features&#x27;: [&#x27;auto&#x27;,
                                                                        &#x27;sqrt&#x27;,
                                                                        &#x27;log2&#x27;,
                                                                        None],
                                        &#x27;decision_tree__min_samples_leaf&#x27;: [1,
                                                                            2,
                                                                            3,
                                                                            4,
                                                                            5,
                                                                            6],
                                        &#x27;decision_tree__min_samples_split&#x27;: array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])},
                   random_state=1042, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-50" type="checkbox" ><label for="sk-estimator-id-50" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">estimator: Pipeline</label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;smote&#x27;, SMOTE(random_state=1042)),
                (&#x27;decision_tree&#x27;, DecisionTreeClassifier(random_state=1042))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-51" type="checkbox" ><label for="sk-estimator-id-51" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">SMOTE</label><div class="sk-toggleable__content fitted"><pre>SMOTE(random_state=1042)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-52" type="checkbox" ><label for="sk-estimator-id-52" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;DecisionTreeClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.tree.DecisionTreeClassifier.html">?<span>Documentation for DecisionTreeClassifier</span></a></label><div class="sk-toggleable__content fitted"><pre>DecisionTreeClassifier(random_state=1042)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
# best parameters 
print('Best parameters for Decision Tree', random_search.best_params_)
```

    Best parameters for Decision Tree {'decision_tree__min_samples_split': 18, 'decision_tree__min_samples_leaf': 3, 'decision_tree__max_features': 'log2', 'decision_tree__max_depth': 5, 'decision_tree__criterion': 'entropy'}
    


```python
# best estimator 
best_estimator = random_search.best_estimator_

# plot
plt.figure(figsize=(20,12))
plot_tree(best_estimator.named_steps['decision_tree'], 
          filled=True, 
         feature_names = train_x.columns.tolist(), 
         class_names = list(map(str, train_y.unique().tolist())))
          
plt.show()
```


    
![png](output_125_0.png)
    



```python
# test the best estimator 
# list that will have all possible models 
models = []

# append models that are going to be used 
models.append(('best decision tree', best_estimator))

models_test(x_reduced, y_reduced, models, train_x, train_y, valid_x, valid_y)
```

    Model best decision tree training Accuracy: 1.0000
    Model best decision tree training Precision: 0.9998
    Model best decision tree training Recall: 0.9998
    Model best decision tree training F1 Score: 0.9998
    Model best decision tree training ROC AUC: 1.0000
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 682611      1
         1      1   4904
    Model best decision tree validation Accuracy: 1.0000
    Model best decision tree validation Precision: 0.9994
    Model best decision tree validation Recall: 0.9991
    Model best decision tree validation F1 Score: 0.9992
    Model best decision tree validation ROC AUC: 0.9995
    
    Confusion Matrix (Accuracy 1.0000)
    
           Prediction
    Actual      0      1
         0 455064      2
         1      3   3276
    Model best decision tree Average cross-validation Accuracy: 0.9999
    Model best decision tree Average cross-validation Precision: 0.9868
    Model best decision tree Average cross-validation Recall: 0.9996
    Model best decision tree Average cross-validation F1 Score: 0.9931
    --------------------------------------------------------------------
    

That is the best parameters that we can get. It is the most explainable while it also has a good performance. The increase in explainability costed only by 1 increase in FP. 

# Final Visualizations

## Confusion Matrix


```python
# threshold classic = 0.5
threshold = 0.5

# true labels
true_labels = []

# predict probabilities 
predicted_probabilities = best_estimator.predict_proba(valid_x)[:, 1]

# predict labels
predicted_labels = (predicted_probabilities > threshold).astype(int)
    
# confusion matrix 
matrix = confusion_matrix(valid_y, predicted_labels)


# heatmap
## figure size (length, width)
plt.figure(figsize=(12, 10))

## heatmap itself
sns.heatmap(matrix, annot=True, fmt='d', 
            xticklabels=['Not Fraudulent', 'Fraudulent'], 
            yticklabels=['Actual Not Fraudulent', 'Actual Fraudulent'])
        
##
colors = ['skyblue', 'grey', 'grey', 'skyblue']

for i, color in enumerate(colors):
    y = i // 2
    x = i % 2
    plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='lightgrey', lw=0.5, color=color))
    
# parameters 
## x  
plt.xlabel('Predicted', fontsize = 16)
plt.xticks(fontsize=16)

## y 
plt.ylabel('Actual', fontsize = 16)
plt.yticks(fontsize=16)


## title
plt.title('Confusion Matrix', fontweight = 'bold', fontsize = 18)

# plot
plt.show()
    
```

    C:\Users\Beibarys Nyussupov\AppData\Local\Temp\ipykernel_35728\3363342237.py:32: UserWarning: Setting the 'color' property will override the edgecolor or facecolor properties.
      plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, fill=True, edgecolor='lightgrey', lw=0.5, color=color))
    


    
![png](output_130_1.png)
    


## Explainable Plot tree




```python
# plot explainable and interpretable decision tree 
# plot
plt.figure(figsize=(20,12))
plot_tree(best_estimator.named_steps['decision_tree'], 
          filled=True, 
          rounded = True, 
          proportion = False, 
          impurity = False, 
        precision = 0,
         feature_names = train_x.columns.tolist(), 
         class_names = list(map(str, train_y.unique().tolist())))


# save the tree 
plt.savefig('decision_tree.png', format='png', bbox_inches='tight', dpi=300)  

plt.show()

```


    
![png](output_132_0.png)
    


## Feature Importance 




```python
# for feature importance 
feature_importance  = pd.Series(best_estimator.named_steps['decision_tree'].feature_importances_, 
                               index = x_reduced.columns).sort_values(ascending = False)


# plot 
sns.barplot(x = feature_importance, 
           y = feature_importance.index)


# parameters 
## title 
plt.title('Feature Importance', fontweight = 'bold', fontsize = 18)

## x
plt.xlabel('Importance', fontsize = 16)
plt.xticks(fontsize = 16)

## y 
plt.ylabel('Features', fontsize = 16)
plt.yticks(fontsize = 16)


plt.show()
```


    
![png](output_134_0.png)
    


## Residual Observations



```python
# predictions 
predictions = best_estimator.predict(valid_x)

# mark predictions which are correct 
correct = predictions == valid_y

## mark incorrect predictions 
incorrect = valid_x[~correct]

# create dataframe for observation 
incorrect['True Labels'] = valid_y[~correct]
incorrect['Predicted Labels'] = predictions[~correct]

# print
pd.DataFrame(incorrect)
```

    C:\Users\Beibarys Nyussupov\AppData\Local\Temp\ipykernel_35728\2819450721.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      incorrect['True Labels'] = valid_y[~correct]
    C:\Users\Beibarys Nyussupov\AppData\Local\Temp\ipykernel_35728\2819450721.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      incorrect['Predicted Labels'] = predictions[~correct]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>type_CASH_OUT</th>
      <th>type_DEBIT</th>
      <th>type_PAYMENT</th>
      <th>type_TRANSFER</th>
      <th>nameDest_Merchant</th>
      <th>True Labels</th>
      <th>Predicted Labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6205439</th>
      <td>353874.22</td>
      <td>353874.22</td>
      <td>353874.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5086130</th>
      <td>29096.95</td>
      <td>29097.16</td>
      <td>0.21</td>
      <td>32876220.28</td>
      <td>32905317.23</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>377151</th>
      <td>42062.82</td>
      <td>340830.43</td>
      <td>298767.61</td>
      <td>398931.35</td>
      <td>678419.64</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>181521</th>
      <td>1041897.02</td>
      <td>1061894.12</td>
      <td>19997.10</td>
      <td>148054.69</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6281484</th>
      <td>399045.08</td>
      <td>10399045.08</td>
      <td>10399045.08</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Final List of variables used:

| Column         | Description                                                          |
|----------------|----------------------------------------------------------------------|
| type           | The type of online transaction.                                      |
| amount         | The amount of the transaction.                                       |
| oldBalanceOrg  | The balance before the transaction of the initiator.                 |
| newBalanceOrig | The balance after the transaction of the initiator.                  |
| nameDest       | The recipient of the transaction.         Merchant/Client            |
| oldBalanceDest | The initial balance of the recipient before the transaction.         |
| newBalanceDest | The new balance of the recipient after the transaction.              |
| isFraud        | Indicates whether the transaction is fraudulent (`Yes` or `No`).     |

# Future Considerations
1. Using all rows without the need to reduce the size of the data - GPUs, additional resources are required to complete the objective 
2. While the decision tree already has a high performance, it is possible to try and use bagging, random forest, and boosting algorithms. 
3. Including more complicated cleaning processes, such as PCA, Outliers Elimination
4. Verifying each transaction in CASHOUT and TRANSFER types by telephone verification message might be a good complementary solution to a decision tree which will improve the safety and performance of the model to higher levels. 
