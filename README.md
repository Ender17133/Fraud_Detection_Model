[![Open in Jupyter](https://img.shields.io/badge/Open%20in-Jupyter-blue.svg?logo=jupyter)](https://github.com/Ender17133/Fraud_Detection_Model/blob/main/Fraud_Detection_Model.ipynb)
[![Open Presentation](https://img.shields.io/badge/Open%20Presentation-Google%20Slides-brightgreen?logo=google-slides)](https://github.com/Ender17133/Fraud_Detection_Model/blob/main/Fraud_Detection_Model.ppt)


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
