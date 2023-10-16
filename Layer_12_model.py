#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


# Read csv data
train = pd.read_csv("dataset_layer12/train.csv")
valid = pd.read_csv("dataset_layer12/valid.csv")


# In[3]:


print("Shape of the train dataset : ", train.shape)
print("Shape of the valid dataset : ", valid.shape)


# In[4]:


# Find out the number of rows with missing values for each column.
def check_missing_values(label):
    print("Missing values in {label}: {val}".format(label=label, val=train[label].isna().sum()))

# Check missing values in labels
for label in ['label_1', 'label_2', 'label_3', 'label_4']:
    check_missing_values(label)
    
# Check for missing values in feature columns
for i in range(1, 773-4):
    if train[label].isna().sum() > 0:
        check_missing_values('feature_' + str(i))
else:
    print("No missing values in feature columns.") 


# In[5]:


# Handling missing values in Label 2
train['label_2'].fillna(train['label_2'].mean().round(), inplace=True)
valid['label_2'].fillna(valid['label_2'].mean().round(), inplace=True)

train = train.astype({'label_2':'int'})
valid = valid.astype({'label_2':'int'})

# Confirm that the values have been filled
train.head()


# In[6]:


# Separate features and target labels
X_train = train.iloc[:, :768]
y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]

X_valid = valid.iloc[:, :768]
y_valid = valid[['label_1', 'label_2', 'label_3', 'label_4']]

# Check the dimensions to confirm
print(X_train.shape, X_valid.shape)
print(y_train.shape, y_valid.shape)


# ## Common functions

# In[7]:


# Remove feature columns with high correlations

def reduce_correlations(corr_matrix, dataset, threshold):    
    removed = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col = corr_matrix.columns[i]
            if abs(corr_matrix.iloc[i, j]) > threshold:
                removed.append(col)
            
    data_filtered = dataset.drop(columns=removed)
    return data_filtered


# In[8]:


# calculate the accuracy of the model

# def evaluate_model(model, x, y):
#     y_pred = model.predict(x)
#     acc = accuracy_score(y, y_pred)
#     return acc

def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy: ", accuracy)
    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix: ")
    print(cm)
    print("Precision, recall, f1-score: ")
    scores = precision_recall_fscore_support(
        y, y_pred, average="weighted")
    return scores, cm


# In[9]:


def PCA_selection(dataset):

    # Standardize the data (important for PCA)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(dataset)

    # Initialize PCA with the number of components you want
    n_components = 0.95  
    pca = PCA(n_components=n_components, svd_solver='full')

    # Fit and transform the data to obtain the principal components
    principal_components = pca.fit_transform(dataset)

    return pca, principal_components


# In[10]:


def cross_validate(x_train, y_train):
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()
    svm = SVC(random_state=42)
    
    f_score_knn = cross_val_score(knn, x_train, y_train, cv = 5)
    print("Cross validation for KNN done. F-Score : ", f_score_knn)
    f_score_rf = cross_val_score(rf, x_train, y_train, cv = 5)
    print("Cross validation for RF done. F-Score : ", f_score_rf)
    f_score_svm = cross_val_score(svm, x_train, y_train, cv = 5)
    print("Cross validation for SVM done. F-Score : ", f_score_svm)
    
    max_f = max(f_score_knn.mean(), f_score_rf.mean(), f_score_svm.mean())
    print("Max mean f-score : ", max_f)
    if f_score_knn.mean() == max_f:
        print("KNN returned")
        return knn
    elif f_score_rf.mean() == max_f:
        print("RF returned")
        return rf
    elif f_score_svm.mean() == max_f:
        print("SVM returned")
        return svm


# In[11]:


def svm_tuner(model, x_train, y_train, x_valid, y_valid):
    
    param_grid = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.1, 1, 10], 'gamma': [0.01,0.001]}
    classifier_model_scores = {}
    
    # Setup random hyperparameter search for model
    rs_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        verbose=2,
        random_state=42,
    )
    
    rs_model.fit(x_train, y_train)
#     classifier_model_scores[name] = rs_model.score(x_valid, y_valid)
#     print("classifier model scores", classifier_model_scores)
    
    return rs_model


# In[12]:


def rf_tuner(model, x_train, y_train, x_valid, y_valid):
    
#         'min_samples_split': [2, 5, 10, 15],
#         'min_samples_leaf': [1, 2, 4, 8],
#         'max_features' : ['auto', 'sqrt'],
    
    param_grid = {
        'n_estimators': np.arange(100, 1000, 100),
        'max_depth': [None] + list(np.arange(10, 110, 10)),
    }
    
    # Setup random hyperparameter search for model
    rs_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=5,
        verbose=2,
        random_state=42,
    )
    
    rs_model.fit(x_train, y_train)
    
    return rs_model


# ## Modeling label 1

# In[13]:


# visualize_data(y_train['label_1'])

series_data_1 = pd.Series(y_train['label_1'])
class_counts_1 = series_data_1.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_1.plot(kind='bar', rot=0)
plt.xlabel('Speaker ID')
plt.ylabel('Count')
plt.title('Distribution of speaker ID')

# Display the bar graph
plt.show()


# In[14]:


# Get correlation matrix for scaled dataset
corr_1 = X_train.corr()


# In[15]:


X_train_filtered_1_1 = reduce_correlations(corr_1, X_train, 0.9)
print(X_train_filtered_1_1.shape)


# In[16]:


# Apply PCA
# Perform PCA on filtered dataset
pca_selector_1, X_train_filtered_1_2 = PCA_selection(X_train_filtered_1_1)
print(X_train_filtered_1_2.shape)


# In[17]:


# Filter the valid dataset as well

X_valid_filtered_1_1 = X_valid[X_train_filtered_1_1.columns]
X_valid_filtered_1_2 = pca_selector_1.transform(X_valid_filtered_1_1)


# In[29]:


# Cross validate and choose best model
model_1_cv = cross_validate(X_train_filtered_1_2, y_train_new['label_1'], 8)


# In[33]:


rs = svm_tuner(model_1_cv, X_train_filtered_1_2, y_train_new['label_1'], X_valid_filtered_1_2, y_valid_new['label_1'])


# In[34]:


# Find the best performing model hyperparameters
rs.best_params_


# In[18]:


from sklearn.model_selection import train_test_split

X_train_new, X_valid_new, y_train_new, y_valid_new = train_test_split(train.iloc[:, :768], train[['label_1', 'label_2', 'label_3', 'label_4']], test_size=0.03, random_state=42)
print(X_train_new.shape, y_train_new.shape)
print(X_valid_new.shape, y_valid_new.shape)


# In[19]:


X_train_1 = X_train_new[X_train_filtered_1_1.columns]
X_train_1_2 = pca_selector_1.transform(X_train_1)

X_valid_1 = X_valid_new[X_train_filtered_1_1.columns]
X_valid_1_2 = pca_selector_1.transform(X_valid_1)


# In[20]:


# Train the model accordingly and evaluate
model_1 = SVC(kernel="rbf", C=1000, gamma=1, random_state=42)
model_1.fit(X_train_1_2, y_train_new['label_1'])


# In[21]:


evaluate_model(model_1, X_valid_1_2, y_valid_new['label_1'])


# ## Modeling label 2

# In[23]:


series_data_2 = pd.Series(y_train['label_2'])
class_counts_2 = series_data_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_2.plot(kind='bar', rot=0)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Speaker Age distribution')

# Display the bar graph
plt.show()


# In[24]:


# Resample data
# Sampling rates calculation

max_count = max(class_counts_2)
print(max_count)
sampling_rates = {}
for c in class_counts_2.keys():
    if class_counts_2[c] > 2000 and class_counts_2[c] < max_count:
        sampling_rates[c] = int(0.95 * max_count)
    elif class_counts_2[c] > 1000 and class_counts_2[c] < max_count:
        sampling_rates[c] = int(0.9 * max_count)
    elif class_counts_2[c] < 1000:
        sampling_rates[c] = int(0.85 * max_count)

print(sampling_rates)

smote_2 = SMOTE(sampling_strategy=sampling_rates, random_state=42)
Xr_train_2, yr_train_2= smote_2.fit_resample(X_train, y_train['label_2'])


# In[26]:


series_data_2_2 = pd.Series(yr_train_2)
class_counts_2_2 = series_data_2_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_2_2.plot(kind='bar', rot=0)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Speaker Age distribution')

# Display the bar graph
plt.show()


# In[28]:


# X_train_filtered_2_1 = reduce_correlations(corr_2, Xr_train, 0.8)
X_train_filtered_2_1 = Xr_train_2[X_train_filtered_1_1.columns]
print(X_train_filtered_2_1.shape)


# In[29]:


# Perform PCA on filtered dataset
pca_selector_2, X_train_filtered_2_2 = PCA_selection(X_train_filtered_2_1)
print(X_train_filtered_2_2.shape)


# In[47]:


# Cross validate and choose best model
model_2_cv = cross_validate(X_train_filtered_2_2, yr_train)


# In[88]:


# X_valid_filtered_2_1 = X_valid[X_train_filtered_2_1.columns]
# X_valid_filtered_2_2 = pca_selector_2.transform(X_valid_filtered_2_1)
# print(X_valid_filtered_2_2.shape)
# print(y_valid_new['label_2'].shape)


# In[30]:


X_train_2, yr_train_2 = smote_2.fit_resample(X_train_new, y_train_new['label_2'])
X_train_2 = X_train_2[X_train_filtered_2_1.columns]
X_train_2_2 = pca_selector_2.transform(X_train_2)

X_valid_2 = X_valid_new[X_train_filtered_2_1.columns]
X_valid_2_2 = pca_selector_2.transform(X_valid_2)


# In[54]:


rs = rf_tuner(model_2_cv, X_train_filtered_2_2, yr_train, X_valid_filtered_2_2, y_valid_new['label_2'])


# In[31]:


# Train the model accordingly and evaluate

model_2 = RandomForestClassifier(n_estimators=200, max_depth=100)
model_2.fit(X_train_2_2, yr_train_2)


# In[57]:


model_2_cv = RandomForestClassifier()
model_2_cv.fit(X_train_2_2, yr_train_2)


# In[32]:


evaluate_model(model_2, X_valid_2_2, y_valid_new['label_2'])


# ## Modeling Label 3

# In[39]:


series_data_3 = pd.Series(y_train_new['label_3'])
class_counts_3 = series_data_3.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_3.plot(kind='bar', rot=0)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Speaker Gender distribution')

# Display the bar graph
plt.show()


# In[40]:


smote_3 = SMOTE(sampling_strategy= 'auto', random_state=42)
Xr_train_3, yr_train_3 = smote_3.fit_resample(X_train_new, y_train_new['label_3'])


# In[42]:


series_data_3_2 = pd.Series(yr_train_3)
class_counts_3_2 = series_data_3_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_3_2.plot(kind='bar', rot=0)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Speaker Gender distribution')

# Display the bar graph
plt.show()


# In[100]:


corr_3 = Xr_train.corr()


# In[44]:


# X_train_filtered_3_1 = reduce_correlations(corr_2, Xr_train, 0.9)
X_train_filtered_3_1 = Xr_train_3[X_train_filtered_1_1.columns]
X_train_filtered_3_1.shape


# In[45]:


# Apply PCA
pca_selector_3, X_train_filtered_3_2 = PCA_selection(X_train_filtered_3_1)
print(X_train_filtered_3_2.shape)


# In[104]:


# Cross validate and choose best model
model_3_cv = cross_validate(X_train_filtered_3_2, yr_train)


# In[47]:


model_3 = RandomForestClassifier()
model_3.fit(X_train_filtered_3_2, yr_train_3)


# In[48]:


# Transform valid dataset
X_valid_filtered_3_1 = X_valid_new[X_train_filtered_3_1.columns]
X_valid_filtered_3_2 = pca_selector_3.transform(X_valid_filtered_3_1)
print(X_valid_filtered_3_2.shape)
print(y_valid_new['label_3'].shape)


# In[49]:


evaluate_model(model_3, X_valid_filtered_3_2, y_valid_new['label_3'])


# In[108]:


rs = rf_tuner(model_2_cv, X_train_filtered_3_2, yr_train, X_valid_filtered_3_2, y_valid_new['label_3'])


# ## Modeling label 4

# In[51]:


series_data_4 = pd.Series(y_train_new['label_4'])
class_counts_4 = series_data_4.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_4.plot(kind='bar', rot=0)
plt.xlabel('Accent')
plt.ylabel('Count')
plt.title('Speaker Accent distribution')

# Display the bar graph
plt.show()


# In[52]:


from imblearn.over_sampling import SMOTE

# Instantiate the SMOTE sampler
smote_4 = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the data
Xr_train_4, yr_train_4 = smote_4.fit_resample(X_train_new, y_train_new['label_4'])


# In[53]:


series_data_4_2 = pd.Series(yr_train_4)
class_counts_4_2 = series_data_4_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_4_2.plot(kind='bar', rot=0)
plt.xlabel('Accent')
plt.ylabel('Count')
plt.title('Speaker Accent distribution')

# Display the bar graph
plt.show()


# In[114]:


corr_4 = Xr_train_4.corr()


# In[54]:


# X_train_filtered_4_1 = reduce_correlations(corr_4, Xr_train_4, 0.8)
X_train_filtered_4_1 = Xr_train_4[X_train_filtered_1_1.columns]
print(X_train_filtered_4_1.shape)


# In[55]:


# Visualize mutual information

mutual_info_4 = mutual_info_classif(X_train_filtered_4_1, yr_train_4)
mutual_info_4 = pd.Series(mutual_info_4)
mutual_info_4.index = X_train_filtered_4_1.columns
mutual_info_4.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[56]:


mi_count_4 = mutual_info_4[mutual_info_4 > 0.08].count()
mi_count_4


# In[57]:


selector4 = SelectKBest(score_func=mutual_info_classif, k=mi_count_4)
X_train_filtered_4_2 = selector4.fit_transform(X_train_filtered_4_1, yr_train_4)
X_train_filtered_4_2.shape


# In[59]:


# Perform PCA on filtered dataset
pca_selector_4, X_train_filtered_4_3 = PCA_selection(X_train_filtered_4_2)
print(X_train_filtered_4_3.shape)


# In[126]:


# Cross validate and choose best model
model_4_cv = cross_validate(X_train_filtered_4_2, yr_train_4)


# In[60]:


model_4 = RandomForestClassifier()
model_4.fit(X_train_filtered_4_3, yr_train_4)


# In[62]:


# Transform valid dataset
X_valid_filtered_4_1 = X_valid_new[X_train_filtered_4_1.columns]
X_valid_filtered_4_2 = selector4.transform(X_valid_filtered_4_1)
X_valid_filtered_4_3 = pca_selector_4.transform(X_valid_filtered_4_2)
print(X_valid_filtered_4_3.shape)
print(y_valid_new['label_4'].shape)


# In[63]:


evaluate_model(model_4, X_valid_filtered_4_3, y_valid_new['label_4'])


# In[137]:


model_4_knn = KNeighborsClassifier()
model_4_knn.fit(X_train_filtered_4_3, yr_train_4)


# In[138]:


evaluate_model(model_4_knn, X_valid_filtered_4_3, y_valid_new['label_4'])


# In[139]:


model_4_svm = SVC()
model_4_svm.fit(X_train_filtered_4_3, yr_train_4)


# In[140]:


evaluate_model(model_4_svm, X_valid_filtered_4_3, y_valid_new['label_4'])


# # Test results

# In[65]:


# Read test data
test = pd.read_csv('dataset_layer12/test.csv')
test.shape


# In[66]:


# Predict the results for all 4 labels
# Preprocess the data according to the 4 models

## LABEL 1
test_filtered_1_1 = test[X_train_filtered_1_1.columns]
test_filtered_1_2 = pca_selector_1.transform(test_filtered_1_1)

## LABEL 2 
test_filtered_2_1 = test[X_train_filtered_2_1.columns]
test_filtered_2_2 = pca_selector_2.transform(test_filtered_2_1)

## LABEL 3
test_filtered_3_1 = test[X_train_filtered_3_1.columns]
test_filtered_3_2 = pca_selector_3.transform(test_filtered_3_1)

## LABEL 4
test_filtered_4_1 = test[X_train_filtered_4_1.columns]
test_filtered_4_2 = selector4.transform(test_filtered_4_1)
test_filtered_4_3 = pca_selector_4.transform(test_filtered_4_2)


# In[67]:


test_1_pred = model_1.predict(test_filtered_1_2)
test_1_pred


# In[68]:


test_2_pred = model_2.predict(test_filtered_2_2)
test_2_pred


# In[69]:


test_3_pred = model_3.predict(test_filtered_3_2)
test_3_pred


# In[70]:


test_4_pred = model_4.predict(test_filtered_4_3)
test_4_pred


# In[71]:


# Write into csv file
import csv

header = ["ID", "label_1", "label_2", "label_3", "label_4"]

with open('190547P_layer12_out.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(744):
        row = [i+1, test_1_pred[i], test_2_pred[i], test_3_pred[i], test_4_pred[i]]
        writer.writerow(row)


# In[ ]:




