#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import dataset
import pandas as pd
student = pd.read_csv('student-por.csv', sep=';')


# In[ ]:


# Binarize data
student = student.get_dummies(student)


# In[ ]:


# Drop school based data
student = student.drop(columns={'G1','G2','school_GP','school_MS','reason_home','reason_course',
                                'reason_reputation','reason_other'})
# Remove rows with zero G3 score
student =  student['G3'] != 0


# In[ ]:


# Splitting the dataset into train, validation, test
from sklearn.model_selection import train_test_split
train, other = train_test_split(student, test_size=0.2, random_state=0)
validation, test = train_test_split(other, test_size=0.5, random_state=0)


# In[ ]:


# Making 'Final Grade' the category to be predicted
X_train = train.drop(columns=['G3'])
y_train = train['G3']

X_val = validation.drop(columns=['G3'])
y_val = validation['G3']
all_columns = X_train.columns


# In[ ]:


# Implementation of the lasso model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def my_fwd_selector(X_train, y_train, X_val, y_val):
    print('=============== Begining forward selection =================')
    cols = list(X_train.columns)
    best_val_acc = 10
    selected_vars = []
    while len(cols) > 0:
        print('Trying {} var models'.format(len(selected_vars) + 1))
        candidate = None
        for i in range(len(cols)):
            current_vars = selected_vars.copy()
            current_vars.append(cols[i])
            if len(current_vars) == 1:
                new_X_train = X_train[current_vars].values.reshape(-1, 1)
                new_X_val = X_val[current_vars].values.reshape(-1, 1)
            else:
                new_X_train = X_train[current_vars]                
                new_X_val = X_val[current_vars]
            
            mod = LogisticRegression(penalty='l1', C=1e9).fit(new_X_train, y_train)
            val_acc = mean_absolute_error(y_val, mod.predict(new_X_val))
            if best_val_acc - val_acc > 0:
                candidate = cols[i]
                best_val_acc = val_acc
        if candidate is not None:
            selected_vars.append(candidate)
            cols.remove(candidate)
            print('------- Adding {} to the model ---------'.format(candidate))
        else:
            break
        print('Columns in current model: {}'.format(', '.join(selected_vars)))
        print('Best mean absolute error: {}'.format(np.round(best_val_acc, 3)))


# In[ ]:


# Algorithm to select which column to add
def select_column_to_add(X_train, y_train, X_val, y_val, columns_in_model, columns_to_test):
    
    column_best = None
    columns_in_model = list(columns_in_model)
    
    if len(columns_in_model) == 0:
        acc_best = 0
    elif len(columns_in_model) == 1:
        mod = LogisticRegression(penalty='l1', C=1e9).fit(X_train[columns_in_model].values.reshape(-1, 1), y_train)
        acc_best = accuracy_score(y_val, mod.predict(X_val[columns_in_model].values.reshape(-1, 1)))
    else:
        mod = LogisticRegression(penalty='l1', C=1e9).fit(X_train[columns_in_model], y_train)
        acc_best = accuracy_score(y_val, mod.predict(X_val[columns_in_model]))

    
    for column in columns_to_test:
        mod = LogisticRegression(penalty='l1', C=1e9).fit(X_train[columns_in_model+[column]], y_train)
        y_pred = mod.predict(X_val[columns_in_model+[column]])
        acc = accuracy_score(y_val, y_pred)
        
        if acc - acc_best > 0:  # one of our stopping criteria
            acc_best = acc
            column_best = column
        
    if column_best is not None:  # the other stopping criteria
        print('Adding {} to the model'.format(column_best))
        print('The new Best mean absolute error is {}'.format(acc_best))
        columns_in_model_updated = columns_in_model + [column_best]
    else:
        print('Did not add anything to the model')
        columns_in_model_updated = columns_in_model
    
    return columns_in_model_updated, acc_best


# In[ ]:


# Print off of the key values
selected_columns = my_fwd_selector(X_train, y_train, X_val, y_val)
y_train_predicted = model.predict(X_train[selected_columns])
y_val_predicted = model.predict(X_val[selected_columns])
model = LogisticRegression(penalty='l1', C=1e9).fit(X_train[selected_columns], y_train)

print('======= Accuracy  table =======')
print('Training mean absolute error is:    {}'.format(mean_absolute_error(y_train, y_train_predicted)))
print('Validation mean absolute error:  {}'.format(mean_absolute_error(y_val, y_val_predicted)))

