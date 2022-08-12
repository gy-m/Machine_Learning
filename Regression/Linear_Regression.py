# Step 1: Import packages and classes
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 2: Provide data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

"""
print("Given x: ", x)
print("Given y: ", y)
"""

# Step 3: Create a model and fit it
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)

# Step 4: Get results
r_sq = model.score(x, y)

"""
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
"""

# Step 5: Predict response
# The created model is used with existing data (x), which was used for creation of the model
y_pred = model.predict(x)
print("For Given x (which used for creation of the model together with y): ", x)
print("The result we wish to get (y): ", y)
print('predicted response (Based on original data x):', y_pred, sep='\n')
print()

# The created model is used with new data (x_new)
x_new = np.arange(5).reshape((-1, 1))
y_pred_new = model.predict(x_new)
print("For x_new (which was not used for creation of the model): ", x_new)
print("The result we wish to get (y_new): Unknown (must predict)")
print('predicted response (Based on new data x_new):', y_pred_new, sep='\n')


