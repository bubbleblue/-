import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

datasets_x = []
datasets_y = []
fr = open('prices.txt')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_x.append(int(items[0]))
    datasets_y.append(int(items[1]))
length =len(datasets_x)
datasets_x = np.array(datasets_x).reshape([length,1])
datasets_y = np.array(datasets_y)

minx = min(datasets_x)
maxx = max(datasets_x)
X = np.arange(minx,maxx).reshape([-1,1])

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_x)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly,datasets_y)

plt.scatter(datasets_x,datasets_y,color='blue')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'green')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()