import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 15)
desired_width = 320
pd.set_option('display.width', desired_width)


def predict(x, theta):
    return np.dot(x, theta)


def theta(x, y):
    return np.dot(np.linalg.pinv(np.dot(x.T, x)), (np.dot(x.T, y)))


train = pd.read_csv('train.csv')
train = train.round(3)
test = pd.read_csv('test.csv')
test = test.round(3)

print(train.corr() > 0.9)
train.drop(['RAD'], axis=1, inplace=True)
test.drop(['RAD'], axis=1, inplace=True)


x_test = test.loc[:, 'CRIM':'LSTAT'].copy().to_numpy()
y_test = test['MEDV'].to_numpy()
x_train = train.loc[:, 'CRIM':'LSTAT'].copy().to_numpy()
y_train = train['MEDV'].to_numpy()

ones = np.ones((x_train.shape[0], 1))
x_train = np.hstack((x_train, ones))
ones = np.ones((x_test.shape[0], 1))
x_test = np.hstack((x_test, ones))

teta = theta(x_train, y_train)
predicted_train = predict(x_train, teta)
predicted_test = predict(x_test, teta)

mse_train = np.linalg.norm(np.subtract(y_train, predicted_train))
mse_test = np.linalg.norm(np.subtract(y_test, predicted_test))

print('MSE train - section.3: {} '.format(mse_train))
print('MSE test - section.3: {} '.format(mse_test))

plt.figure()
plt.scatter(y_train, predicted_train)
plt.scatter(y_test, predicted_test, color='red')
plt.xlabel('y')
plt.ylabel('predict')
plt.title('train points: blue\n test points: red')
plt.show()

plt.figure()
plt.scatter(test['LSTAT'], y_test)
plt.scatter(test['LSTAT'], predicted_test, color='red')
plt.xlabel('LSTAT')
plt.ylabel('y')
plt.show()

plt.figure()
plt.scatter(train['B'], y_train)
plt.scatter(train['B'], predicted_train, color='red')
plt.xlabel('B')
plt.ylabel('y')
plt.show()
#
plt.figure()
plt.scatter(train['DIS'], y_train)
plt.scatter(train['DIS'], predicted_train, color='red')
plt.xlabel('DIS')
plt.ylabel('y')
plt.show()

plt.figure()
plt.scatter(train['CHAS'], y_train)
plt.scatter(train['CHAS'], predicted_train, color='red')
plt.xlabel('CHAS')
plt.ylabel('y')
plt.show()

plt.figure()
plt.scatter(train['TAX'], y_train)
plt.scatter(train['TAX'], predicted_train, color='red')
plt.xlabel('TAX')
plt.ylabel('y')
plt.show()

plt.figure()
plt.scatter(train['AGE'], y_train)
plt.scatter(train['AGE'], predicted_train, color='red')
plt.xlabel('AGE')
plt.ylabel('y')
plt.show()
