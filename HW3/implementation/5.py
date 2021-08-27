import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rd
pd.set_option('display.max_columns', 15)
desired_width = 320
pd.set_option('display.width', desired_width)


def gaussian(x, i, j):
    dist = np.linalg.norm(np.subtract(x[i], x[j]))
    return np.exp(-(dist**2) / 2)

def predict(x, theta):
    return np.dot(x, theta)


def theta(x, y):
    tmp1 = np.dot(np.transpose(x), x)
    tmp2 = np.dot(np.transpose(x), y)
    return np.dot(np.linalg.pinv(tmp1), tmp2)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.round(3)
test = test.round(3)
train.drop(['RAD'], axis=1, inplace=True)
test.drop(['RAD'], axis=1, inplace=True)

y_train = train['MEDV'].to_numpy()
x_train = train.loc[:, 'CRIM':'LSTAT'].copy().to_numpy()
y_test = test['MEDV'].to_numpy()
x_test = test.loc[:, 'CRIM':'LSTAT'].copy().to_numpy()

train_pivots = rd.sample([i for i in range(x_train.shape[0])], 10)
test_pivots = rd.sample([i for i in range(x_test.shape[0])], 10)

x_train_gaussian, x_test_gaussian = np.ones((x_train.shape[0], 1)), np.ones((x_test.shape[0], 1))
for pivot in train_pivots:
        tmp = []
        for i in range(x_train.shape[0]):
            tmp.append(gaussian(x_train, i, pivot))
        x_train_gaussian = np.hstack((x_train_gaussian, np.asarray([tmp]).T))

for pivot in test_pivots:
        tmp = []
        for i in range(x_test.shape[0]):
            tmp.append(gaussian(x_test, i, pivot))
        x_test_gaussian = np.hstack((x_test_gaussian, np.asarray([tmp]).T))

x_train_gaussian = np.hstack((x_train_gaussian, x_train))
x_test_gaussian = np.hstack((x_test_gaussian, x_test))

print(x_train_gaussian.shape)
print(x_test_gaussian.shape)

teta = theta(x_train_gaussian, y_train)
predicted_train = predict(x_train_gaussian, teta)
predicted_test = predict(x_test_gaussian, teta)

mse_train = np.linalg.norm(np.subtract(y_train, predicted_train))
mse_test = np.linalg.norm(np.subtract(y_test, predicted_test))

print('MSE train - section.5: {} '.format(mse_train))
print('MSE test - section.5: {} '.format(mse_test))

plt.figure()
plt.scatter(y_train, predicted_train)
plt.scatter(y_test, predicted_test, color='red')
plt.xlabel('y')
plt.ylabel('predict')
plt.title('train points: blue\n test points: red')
plt.show()




plt.figure()
plt.scatter(train['LSTAT'], y_train)
plt.scatter(train['LSTAT'], predicted_train, color='red')
plt.xlabel('LSTAT')
plt.ylabel('y')
plt.show()

plt.figure()
plt.scatter(train['B'], y_train)
plt.scatter(train['B'], predicted_train, color='red')
plt.xlabel('B')
plt.ylabel('y')
plt.show()

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
