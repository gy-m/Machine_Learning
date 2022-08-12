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
#Select single value by row
# print(df.iloc[0])

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

# Cheking the scores
    # In this case we can see that the logistic regression reached an impressive 79.6%
# print(score)
    # check with each data set we have
for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy for {} data: {:.4f}'.format(source, score))


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

# configure the learning process using compile method
# compile method specifies the optimizer (method to find the best weights) and the loss function (method to calculate the error)
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

# training process
# clear session must be done only before running the fit once again. It is not suppose to be cleared in the first time
# clear_session()
history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

# evaluate the accuracy of the model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))                    


"""
# visualize the loss and accuracy for the training and testing data based on the History callback
# make sure your python version is supported (python --version) and if not just uninstall it (conda uninstall python) and install a supported version
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

# The plot can be presented during debugging (typing this command in the debug console)
# plot_history(history)
"""