# Customer_Credit_Risk
A Machine Learning project aiming at creating a comprehensive evaluation system for customer credit &amp; risk management.

Xinyi-Glass is a glass manufacturing company having customers from diverse industries. After Xinyi shipped the glass product to the customer, there is a period (based on the scale and credibility of the customer) that the customer must pay the money to prevent overdue. This model is used to give a credibility score for each customer based on various related metrics. The score will be a reference for the salesperson when assigning credit limits and days to new customers, as well as altering credit limits for existing customers.

Here are the steps for model development:

1. ETL in Hive database for data extraction and pre-processing
2. Data exploratory analysis & Data Engineering (SQL)
3. Feature Engineering: add concerned variables that could be related to abnormal oiling behaviors (SQL)
4. Use a decision tree model to determine the boundaries for woe(weight of evidence) encoding
5. Calculate the IV value for each variable, then eliminate variables with IV less than 0.2
6. Fit the data with Logistic Regression
7. Use the logistic regression result, woe weights to convert into scores
8. Save the model for future use
9. Set up the code scheduling to run the model once a month

