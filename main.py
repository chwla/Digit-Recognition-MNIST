from sklearn.datasets import fetch_openml
import  matplotlib.pyplot as mp
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

mnist = fetch_openml('mnist_784')

x, y = mnist['data'], mnist['target']

digit = x.to_numpy()[30000]
digit_image = digit.reshape(28, 28)
mp.imshow(digit_image, cmap='binary', interpolation='nearest')
mp.axis('off')
mp.show()

y = y.astype(int)

indices = np.arange(len(x))
np.random.shuffle(indices)

split = 0.8
split_index = int(len(indices) * split)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

x = np.array(x)
y = np.array(y)

x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]

a = LogisticRegression(C=0.1, tol=0.1, max_iter=1000, solver='lbfgs')
a.fit(x_train, y_train)

ans = a.predict([digit])
print(f"Predicted digit: {ans[0]}")

cv_score = cross_val_score(a, x_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_score}")
print(f"Mean cross-validation score: {np.mean(cv_score)}")

accuracy = np.mean(a.predict(x_test) == y_test)
print(f"Accuracy on test set: {accuracy}")

