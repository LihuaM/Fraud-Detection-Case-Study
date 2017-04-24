# Fraud-Detection-Case-Study

* An e-commerce site tries to weed out fraudsters.
* The dataset originally contains 14,337 data records with 43 features.
* Highly imbalanced dataset with 91% not fraud data records.
* Business Goal: detect if an event is a fraud or not.

After EDA, we decided to keep the following features into our final models:

* delivery_method
* num_payouts
* num_order
* org_facebook
* org_twitter
* sale_duration
* previous_payouts_total (hand crafted feature)
* duration_days (hand crafted feature)

We split our dataset into training dataset (0.9) and testing dataset(0.1), the following results were based on our testing dataset using Random Forest Algorithm.

* Accuracy Score:  0.99
* Recall Score:  0.90
* Precision Score:  0.91
* AUC Score:  0.98
