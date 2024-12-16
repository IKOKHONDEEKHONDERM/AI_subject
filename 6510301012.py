import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd

url = r"D:\WORK\Year-3-2\AI\titanic.csv"  # Use raw string for Windows path
data = pd.read_csv(url)

# make_blobs
X, y = make_blobs(n_samples=200, centers=[[2.0, 2.0], [3.0, 3.0]], cluster_std=0.75, random_state=42)
# train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Build model
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=300, batch_size=200, verbose=0)

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.round(Z).astype(int)
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k', cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Decision Boundary')
plt.legend(['Class 1', 'Class 2'])
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# import pandas as pd

# url = r"D:\WORK\Year-3-2\AI\titanic.csv"  # Use raw string for Windows path
# data = pd.read_csv(url)

# # Prepare data
# X = data[['Age', 'Fare']].values
# y = data['Survived'].values
# X[:, 0] = np.nan_to_num(X[:, 0], nan=np.nanmin(X[:, 0]))
# X[:, 1] = np.nan_to_num(X[:, 1], nan=np.nanmin(X[:, 1]))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# # Build model
# model = Sequential()
# model.add(Dense(16, input_dim=2, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# optimizer = Adam(0.0001)
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# # Train model
# model.fit(X_train, y_train, epochs=300, batch_size=200, verbose=0)

# # Predictions
# y_pred_prob = model.predict(X_test)
# y_pred = np.round(y_pred_prob).astype(int).ravel()

# # Decision boundary
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = np.round(Z).astype(int)
# Z = Z.reshape(xx.shape)

# # Plot
# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.title('Decision Boundary')
# plt.show()
