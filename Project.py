# Importing the Dependencies

import numpy as np  # used to create numpy arrays
import pandas as pd  # used to create dataframe (structural data)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# the text data to numerical values(meaningful values)-- i.e. feature vector

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the data from csv file to a pandas dataframe
raw_mail_data = pd.read_csv('mail_data.csv')

# replace the null values with a null strings(empty string)
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0

# In this mail data frame we are locating the mail data values(span) category as 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# separating the data as text and label (x axis and y axis respectively)
X = mail_data['Message']
Y = mail_data['Category']

# printing the type of data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#      ----- arrays -------
# random_state -- will split the data training and test data in different manners


# transform the text data to feature vectors can be used as input to the logisti regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')

# TfidfVectorizer -- It will give some score or weight score according to the words and linked to spam and ham mail
# min df -- score < 1 -- exclude otherwise include
# stop_words -- multiple time repetitive is the extra words -- will be excluded
# lowercase -- all the letter will be converted to lower case

X_train_features = feature_extraction.fit_transform(X_train)
# fitting the data Vectorizer and transforming them into numerical values -- meaningful

X_test_features = feature_extraction.transform(X_test)
# not to look at X_test that's why not fitting it, just transforming it

# convert Y_train and Y_test as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Training the model

model = LogisticRegression()
# training the Logistic Regression model with the training data

model.fit(X_train_features, Y_train)

# Evaluating the model

# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

# 96% i.e. 96 out of 100 is correct prediction

# prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# model may over fit i.e. model is well performing on training data but not on test data
# over-trained from the data -- difference b/w the train and test prediction


def prediction_model(text):
    input_mail = text
    # ham should give the value as 1
    # convert test to feature vectors
    input_data_features = feature_extraction.transform(input_mail)
    # making prediction
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        return 'Ham mail'
    else:
        return 'Spam mail'

