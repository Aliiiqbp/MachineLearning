import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston

dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target
print(df.isnull().sum())

plt.hist(df['MEDV'], bins=75)
plt.gca().set(title='MEDV Distribution', ylabel='Frequency')
plt.show()

print(len(df[df['MEDV'] == df['MEDV'].max()]))
df = df[df['MEDV'] != df['MEDV'].max()]

train = df.sample(frac=0.8, random_state=int(time.time()), replace=False)
test = df.drop(train.index)
print(train.shape, test.shape)

train.to_csv(r'train.csv', index=False, header=True)
test.to_csv(r'test.csv', index=False, header=True)
