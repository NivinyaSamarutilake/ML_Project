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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# In[2]:


# Read csv data
train = pd.read_csv("dataset_layer7/train.csv")
valid = pd.read_csv("dataset_layer7/valid.csv")


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


# ## Common Functions
# 
# In this section, the functions which are reused throughout the notebook are included.

# In[8]:


# cross validation function
# Here, the cross validation will be done across these models -> KNN, RandomForest and SVM
# 
from sklearn.model_selection import KFold, cross_val_score

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


# In[9]:


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


# In[10]:


# calculate the accuracy of the model

def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred)
    return acc


# In[11]:


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
    


# In[12]:


from sklearn.model_selection import RandomizedSearchCV
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


# In[13]:


def knn_tuner(model, x_train, y_train, x_valid, y_valid):
    
    param_grid = {'n_neighbors': list(range(1,15)),'algorithm': ('auto', 'ball_tree', 'kd_tree' , 'brute') }
    
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
    


# ## Modelling label_1 : Speaker ID

# In[14]:


# visualize_data(y_train['label_1'])

series_data = pd.Series(y_train['label_1'])
class_counts = series_data.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts.plot(kind='bar', rot=0)
plt.xlabel('Speaker ID')
plt.ylabel('Count')
plt.title('Distribution of speaker ID')

# Display the bar graph
plt.show()


# In[14]:


# Get the correlation matrix for speakerID labels
corr_1 = X_train.corr()


# In[15]:


X_train_filtered_1_1 = reduce_correlations(corr_1, X_train, 0.6)
print(X_train_filtered_1_1.shape)


# In[16]:


# Visualize mutual information

mutual_info_1 = mutual_info_classif(X_train_filtered_1_1, y_train['label_1'])
mutual_info_1 = pd.Series(mutual_info_1)
mutual_info_1.index = X_train_filtered_1_1.columns
mutual_info_1.sort_values(ascending=False).plot.bar(figsize=(20, 8))

mi_count_1 = mutual_info_1[mutual_info_1 > 0.065].count()


# In[17]:


mi_count_1


# In[18]:


selector1 = SelectKBest(score_func=mutual_info_classif, k=mi_count_1)
X_train_filtered_1_2 = selector1.fit_transform(X_train_filtered_1_1, y_train['label_1'])
X_train_filtered_1_2.shape


# In[19]:


# Perform PCA on filtered dataset
print(X_train_filtered_1_2.shape)
pca_selector_1, X_train_filtered_1_3 = PCA_selection(X_train_filtered_1_2)
print(X_train_filtered_1_3.shape)


# ### Cross validation
# 
# Find the best fitting model with cross validation. The considered models are KNeighborsClassifier, Random Forest Classifier and the Support Vector Machine. 

# In[31]:


model_1_cv = cross_validate(X_train_filtered_1_3, y_train['label_1'])


# In[37]:


rs = svm_tuner(model_1_cv, X_train_filtered_1_3, y_train['label_1'], X_valid_filtered_1_3, y_valid['label_1'])


# In[38]:


rs.best_params_


# In[20]:


# Train the model accordingly and evaluate
model_1 = SVC(kernel="linear", C=1, gamma=0.01)
model_1.fit(X_train_filtered_1_3, y_train['label_1'])


# In[21]:


X_valid_filtered_1_1 = X_valid[X_train_filtered_1_1.columns]
X_valid_filtered_1_2 = selector1.transform(X_valid_filtered_1_1)
X_valid_filtered_1_3 = pca_selector_1.transform(X_valid_filtered_1_2)


# In[22]:


evaluate_model(model_1, X_valid_filtered_1_3, y_valid['label_1'])


# Through manual hyperparameter tuning the following model was found to perform better than the above selected one.

# In[23]:


# Train a KNN model on the filtered dataset
model_1 = KNeighborsClassifier(n_neighbors = 4)
model_1.fit(X_train_filtered_1_3, y_train['label_1'])


# In[24]:


# Evaluate the model's performance
# X_valid_filtered_1_1 = X_valid[X_train_filtered_1_1.columns]
# X_valid_filtered_1_2 = selector1.transform(X_valid_filtered_1_1)
# X_valid_filtered_1_3 = pca_selector_1.transform(X_valid_filtered_1_2)
evaluate_model(model_1, X_valid_filtered_1_3, y_valid['label_1'])


# In[25]:


# get predictions for train and valid datasets

train_pred_1 = model_1.predict(X_train_filtered_1_3)
valid_pred_1 = model_1.predict(X_valid_filtered_1_3)

pd_train_pred_1 = pd.DataFrame({'label_1':train_pred_1})
pd_train_pred_1.to_csv('layer7_train_label_1.csv', index=True)

pd_valid_pred_1 = pd.DataFrame({'label_1':valid_pred_1})
pd_valid_pred_1.to_csv('layer7_valid_label_1.csv', index=True)


# ### Modelling label_2 : Speaker Age

# In[26]:


series_data_2 = pd.Series(y_train['label_2'])
class_counts_2 = series_data_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_2.plot(kind='bar', rot=0)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Speaker Age distribution')

# Display the bar graph
plt.show()


# In[27]:


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
Xr_train, yr_train= smote_2.fit_resample(X_train, y_train['label_2'])


# In[28]:


series_data_2_2 = pd.Series(yr_train)
class_counts_2_2 = series_data_2_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_2_2.plot(kind='bar', rot=0)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Speaker Age distribution')

# Display the bar graph
plt.show()


# Now that the classes are balanced, let's start with removing features. We have to first find the correlation matrix of X_train_resampled and remove highly correlated featues. Then we have to remove mutual info. Then PCA

# In[29]:


corr_2 = Xr_train.corr()


# In[30]:


X_train_filtered_2_1 = reduce_correlations(corr_2, Xr_train, 0.6)
print(X_train_filtered_2_1.shape)


# In[31]:


# Visualize mutual information

mutual_info_2 = mutual_info_classif(X_train_filtered_2_1, yr_train)
mutual_info_2 = pd.Series(mutual_info_2)
mutual_info_2.index = X_train_filtered_2_1.columns
mutual_info_2.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[34]:


mi_count_2 = mutual_info_2[mutual_info_2 > 0.06].count()
mi_count_2


# In[35]:


selector2 = SelectKBest(score_func=mutual_info_classif, k=mi_count_2)

X_train_filtered_2_2 = selector2.fit_transform(X_train_filtered_2_1, yr_train)
X_train_filtered_2_2.shape


# In[36]:


# Perform PCA on filtered dataset
# pca_selector_2, X_train_filtered_2_3 = PCA_selection(X_train_filtered_2_2)
# print(X_train_filtered_2_3.shape)


# In[41]:


# Perform PCA on filtered dataset
pca_selector_2, X_train_filtered_2_3 = PCA_selection(X_train_filtered_2_1)
print(X_train_filtered_2_3.shape)


# ### Cross Validation

# In[42]:


model_2_cv = cross_validate(X_train_filtered_2_3, yr_train)


# In[42]:


X_valid_filtered_2_1 = X_valid[X_train_filtered_2_1.columns]
X_valid_filtered_2_3 = pca_selector_2.transform(X_valid_filtered_2_1)
print(X_valid_filtered_2_3.shape)
print(y_valid['label_2'].shape)


# In[60]:


rs2 = svm_tuner(model_2_cv, X_train_filtered_2_3, yr_train, X_valid_filtered_2_3, y_valid['label_2'])


# In[44]:


model_2 = SVC(random_state=42)
model_2.fit(X_train_filtered_2_3, yr_train)


# In[39]:


# X_valid_filtered_2_1 = X_valid[X_train_filtered_2_1.columns]
# X_valid_filtered_2_2 = selector2.transform(X_valid_filtered_2_1)
# X_valid_filtered_2_3 = pca_selector_2.transform(X_valid_filtered_2_2)
# print(X_valid_filtered_2_3.shape)
# print(y_valid['label_2'].shape)


# In[45]:


evaluate_model(model_2, X_valid_filtered_2_3, y_valid['label_2'])


# In[37]:


# model_2_cv = SVC(random_state=42)
# model_2_cv.fit(X_train_filtered_2_3, yr_train)


# ## Modeling label_3 : Speaker Gender

# In[47]:


series_data_3 = pd.Series(y_train['label_3'])
class_counts_3 = series_data_3.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_3.plot(kind='bar', rot=0)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Speaker Gender distribution')

# Display the bar graph
plt.show()


# There is class imbalance. Let's oversample.

# In[48]:


oversampler3 = RandomOverSampler(sampling_strategy= 0.85, random_state=42)
Xr_train, yr_train = oversampler3.fit_resample(X_train, y_train['label_3'])


# In[49]:


series_data_3_2 = pd.Series(yr_train)
class_counts_3_2 = series_data_3_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_3_2.plot(kind='bar', rot=0)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Speaker Gender distribution')

# Display the bar graph
plt.show()


# In[50]:


corr_3 = Xr_train.corr()


# In[51]:


X_train_filtered_3_1 = reduce_correlations(corr_3, Xr_train, 0.6)
print(X_train_filtered_3_1.shape)


# In[53]:


selector3 = SelectKBest(score_func=mutual_info_classif, k=150)
X_train_filtered_3_2 = selector3.fit_transform(X_train_filtered_3_1, yr_train)
X_train_filtered_3_2.shape


# In[54]:


# Perform PCA on filtered dataset
pca_selector_3, X_train_filtered_3_3 = PCA_selection(X_train_filtered_3_2)
print(X_train_filtered_3_3.shape)


# In[70]:


# Hyperparameter tuning and cross validation
from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train_filtered_3_3, yr_train)

param_grid = {'n_neighbors': list(range(1,15))}
gs = GridSearchCV(knn_clf, param_grid, cv=10)
gs.fit(X_train_filtered_3_3, yr_train)

print(gs.best_params_)


# In[69]:


gs.cv_results_['params']


# In[55]:


# Train a KNN model on the filtered dataset
model_3 = KNeighborsClassifier(n_neighbors = 3)
model_3.fit(X_train_filtered_3_3, yr_train)


# In[56]:


# Oversample the valid dataset
Xr_valid_3, yr_valid_3 = oversampler3.fit_resample(X_valid, y_valid['label_3']) 
X_valid_filtered_3_1 = Xr_valid_3[X_train_filtered_3_1.columns]
X_valid_filtered_3_2 = selector3.transform(X_valid_filtered_3_1)
X_valid_filtered_3_3 = pca_selector_3.transform(X_valid_filtered_3_2)
print(evaluate_model(model_3, X_valid_filtered_3_3, yr_valid_3))


# ## Modeling label_4 : Speaker Accent

# In[14]:


series_data_4 = pd.Series(y_train['label_4'])
class_counts_4 = series_data_4.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_4.plot(kind='bar', rot=0)
plt.xlabel('Accent')
plt.ylabel('Count')
plt.title('Speaker Accent distribution')

# Display the bar graph
plt.show()


# In[15]:


from imblearn.over_sampling import SMOTE

# Define the desired number of samples per class (e.g., equal to the count of the majority class)
# desired_count = class_counts.mean()

# Instantiate the SMOTE sampler
smote_4 = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the data
Xr_train_4, yr_train_4 = smote_4.fit_resample(X_train, y_train['label_4'])


# In[16]:


series_data_4_2 = pd.Series(yr_train_4)
class_counts_4_2 = series_data_4_2.value_counts()

plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
class_counts_4_2.plot(kind='bar', rot=0)
plt.xlabel('Accent')
plt.ylabel('Count')
plt.title('Speaker Accent distribution')

# Display the bar graph
plt.show()


# In[17]:


corr_4 = Xr_train_4.corr()


# In[18]:


X_train_filtered_4_1 = reduce_correlations(corr_4, Xr_train_4, 0.6)
print(X_train_filtered_4_1.shape)


# In[19]:


selector4 = SelectKBest(score_func=mutual_info_classif, k=300)
X_train_filtered_4_2 = selector4.fit_transform(X_train_filtered_4_1, yr_train_4)
X_train_filtered_4_2.shape


# In[20]:


# Perform PCA on filtered dataset
pca_selector_4, X_train_filtered_4_3 = PCA_selection(X_train_filtered_4_2)
print(X_train_filtered_4_3.shape)


# In[ ]:


model_4_cv = cross_validate(X_train_filtered_4_3, yr_train_4)


# In[21]:


# Train a KNN model on the filtered dataset
model_4 = KNeighborsClassifier(n_neighbors = 4)
model_4.fit(X_train_filtered_4_3, yr_train_4)


# In[23]:


# Oversample the valid dataset
Xr_valid_4, yr_valid_4 = smote_4.fit_resample(X_valid, y_valid['label_4']) 
X_valid_filtered_4_1 = Xr_valid_4[X_train_filtered_4_1.columns]
X_valid_filtered_4_2 = selector4.transform(X_valid_filtered_4_1)
X_valid_filtered_4_3 = pca_selector_4.transform(X_valid_filtered_4_2)
print(evaluate_model(model_4, X_valid_filtered_4_3, yr_valid_4))


# In[116]:


from sklearn.model_selection import GridSearchCV
knn_clf_2 = KNeighborsClassifier()
# knn_clf_2.fit(X_train_filtered_7, y_resampled_4)

param_grid = {'n_neighbors': list(range(1,15))}
gs = GridSearchCV(knn_clf_2,param_grid,cv=10)
gs.fit(X_valid_filtered_4_3, yr_valid_4)


# In[117]:


gs.best_params_


# # Testing the models

# In[26]:


# Read test data
test = pd.read_csv('dataset_layer7/test.csv')
test.shape


# In[83]:


# Predict the results for all 4 labels
# Preprocess the data according to the 4 models

## LABEL 1
test_filtered_1_1 = test[X_train_filtered_1_1.columns]
test_filtered_1_2 = selector1.transform(test_filtered_1_1)
test_filtered_1_3 = pca_selector_1.transform(test_filtered_1_2)

## LABEL 2 
test_filtered_2_1 = test[X_train_filtered_2_1.columns]
# test_filtered_2_2 = selector2.transform(test_filtered_2_1)
test_filtered_2_3 = pca_selector_2.transform(test_filtered_2_1)

## LABEL 3
test_filtered_3_1 = test[X_train_filtered_3_1.columns]
test_filtered_3_2 = selector3.transform(test_filtered_3_1)
test_filtered_3_3 = pca_selector_3.transform(test_filtered_3_2)

## LABEL 4
test_filtered_4_1 = test[X_train_filtered_4_1.columns]
test_filtered_4_2 = selector4.transform(test_filtered_4_1)
test_filtered_4_3 = pca_selector_4.transform(test_filtered_4_2)


# In[84]:


test_1_pred = model_1.predict(test_filtered_1_3)
test_1_pred


# In[85]:


test_2_pred = model_2.predict(test_filtered_2_3)
test_2_pred


# In[86]:


test_3_pred = model_3.predict(test_filtered_3_3)
test_3_pred


# In[87]:


test_4_pred = model_4.predict(test_filtered_4_3)
test_4_pred


# In[91]:


# Write into csv file
import csv

header = ["ID", "label_1", "label_2", "label_3", "label_4"]

with open('190547P_layer7_out.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(744):
        row = [i+1, test_1_pred[i], test_2_pred[i], test_3_pred[i], test_4_pred[i]]
        writer.writerow(row)
        


# In[ ]:




