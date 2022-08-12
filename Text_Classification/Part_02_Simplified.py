import pandas as pd

############################ Step 1 - Creating a Data Set ############################

# dict which represents the source and the path of that source
filepath_dict = {'yelp':   'Text_Classification\\data\\yelp_labelled.txt',
                 'amazon': 'Text_Classification\\data\\amazon_cells_labelled.txt',
                 'imdb':   'Text_Classification\\data\\imdb_labelled.txt'}


# data folder list - holds all 3 tables
df_list = []
for source, filepath in filepath_dict.items():
    # read the txt and transform it to csv with two colums, which will
    # be seperated with tab
    # df is a pandas object named dataframe
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    # Add another column filled with the source name
    df['source'] = source  
    df_list.append(df)

# df_list is the list with all 3 tables
# df is defined to be one big table (pandas object named dataframe) with all the data from all the tables
df = pd.concat(df_list)

# Printing the data set
print("\n\nData Set: ", df)


############################ Step 2 - Creating a Baseline Model ############################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Selecting "yelp" table section
df_yelp = df[df['source'] == 'yelp']

# selecting the sentences (col) of "yelp" table section (we get ndarray object of numpy, instead of pandas)
sentences = df_yelp['sentence'].values
# selecting the label (col) of "yelp" table section (we get ndarray object of numpy, instead of pandas)
y = df_yelp['label'].values

# train_test_split is used for spliting cleverly the data set (sentences) into training and testing sets
# in addition it gives us the feature vectors of training and testings sets
# The reason we use this function, instead spliting the set ourself, beacuse we may split it with a bad distribution
# for example: We may create a training set with only positive feedback and testing set only with negative feedback
# sentences_train, sentences_test are the sentences which will be used for training and testing
# y_train and y_test are the labels of the sentences
# test_size - the precentage of the set to be used for testing (usually around 20%-30%)
# random_state  - number of times for training
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Vectorizing the sentences using BOW model 
    # creating a vocabulary
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
    # creating a feature vectors for train and test data set
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# Choosing baseline model 
# In other words this is our classification model - logistic regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

# printing the score of the baseline model (LogisticRegression)
print("\n\nThe score of LogisticRegression (Baseline) model (Non Deep Neural Model): ", score, "\n\n")

############################ Step 3 - Creating a Neural Model ############################
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session

# Getting  the number of dimensions of the feature vectors (Number of features)
input_dim = X_train.shape[1]  

# Creating a model of type Sequential
model = Sequential()
# Adding 2 Layers and assigning activation functions
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configure the learning process using compile method
# Compile method specifies the optimizer (method to find the best weights) and the loss function (method to calculate the error)
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Printing the Summary of the trained Sequential model
print("\n\nSummary of the trained Sequential model (Deep Neural Model):\n")
model.summary()

# Training process
# clear session must be done only before running the fit once again. It is not suppose to be cleared in the first time
# clear_session()
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

# evaluate the accuracy of the model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("\n\nTraining Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy), "\n\n")                    