from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Preparing the testing and training sets
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

############################## Linear Regression example ##############################

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)
# It is not a good practive to rely on training set for the score
print("Linear Regression - Score based on training sets: ", model.score(x_train, y_train))
# One should rely on the testing set for the score
print("Linear Regression - Score based on testing sets: ", model.score(x_test, y_test))

############################## Gradient boosting Regression example ##############################

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
# It is not a good practive to rely on training set for the score
print("Gradient boosting Regression - Score based on training sets: ", model.score(x_train, y_train))
# One should rely on the testing set for the score
print("Gradient boosting Regression - Score based on testing sets: ", model.score(x_test, y_test))

############################## Random forest Regression example ##############################

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
# It is not a good practive to rely on training set for the score
print("Random forest Regression - Score based on training sets: ", model.score(x_train, y_train))
# One should rely on the testing set for the score
print("Random forest Regression - Score based on testing sets: ", model.score(x_test, y_test))
