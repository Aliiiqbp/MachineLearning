import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 15)
desired_width = 320
pd.set_option('display.width', desired_width)


df = pd.read_csv('train.csv')
print(df.corr().round(2))
print('\n')
print(df.describe().round(2))

#LSTAT
plt.scatter(df['MEDV'], df['LSTAT'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('LSTAT')
plt.show()

#B
plt.scatter(df['MEDV'], df['B'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('B')
plt.show()

#PTRATIO
plt.scatter(df['MEDV'], df['PTRATIO'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('PTRATIO')
plt.show()

#TAX
plt.scatter(df['MEDV'], df['TAX'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('TAX')
plt.show()

#RAD
plt.scatter(df['MEDV'], df['RAD'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('RAD')
plt.show()

#DIS
plt.scatter(df['MEDV'], df['DIS'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('DIS')
plt.show()

#AGE
plt.scatter(df['MEDV'], df['AGE'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('AGE')
plt.show()

#RM
plt.scatter(df['MEDV'], df['RM'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('RM')
plt.show()

#NOX
plt.scatter(df['MEDV'], df['NOX'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('NOX')
plt.show()

#CHAS
plt.scatter(df['MEDV'], df['CHAS'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('CHAS')
plt.show()

#INDUS
plt.scatter(df['MEDV'], df['INDUS'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('INDUS')
plt.show()

#ZN
plt.scatter(df['MEDV'], df['ZN'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('ZN')
plt.show()

#CRIM
plt.scatter(df['MEDV'], df['CRIM'])
plt.gca().set(title='Correlation with "MEDV"')
plt.xlabel('MEDV')
plt.ylabel('CRIM')
plt.show()
