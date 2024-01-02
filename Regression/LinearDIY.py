from statistics import mean
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

import random

style.use("fivethirtyeight")
df= pd.read_csv("./data.csv")
# print(df)
# xs=[x for x in range(1,7)]
# ys=[5,4,6,5,6,7]

# xs=np.array(xs,dtype=np.float64)
# ys=np.array(ys,dtype=np.float64)

def best_fit_params(xs,ys)->int:
    m=(mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)**2)-mean(xs*xs))
    b= mean(ys)-mean(xs)*m
    return m,b

def sqaured_error(ys_og,ys_lines):
    return sum((ys_lines-ys_og)**2)

#sqaured error is used to find out the error between the recieved line and the actual line with the square being given as a penalty for outliers
def coef_det(ys_og,ys_lines)->float:
    ys_mean_line=[mean(ys_og) for y in ys_og]
    print(f"mean line is:\n{ys_mean_line}\nregression line is:\n{ys_lines}")
    squared_error_regression=sqaured_error(ys_og,ys_lines)
    print(f"regression: {squared_error_regression}")
    squared_error_mean=sqaured_error(ys_og,ys_mean_line)
    print(f"mean is {squared_error_mean}")
    return float(1-(squared_error_regression/squared_error_mean))
#the coeff of determination determines how good of a line we were able to create by testing it against the worst case scecario of errors and if r^2 is greater then the
#model is that much better

def create_dataset(hm,variance,step=2,correlation=False):
    val=1
    ys=[]
    for i in range(hm):
        y = val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='neg':
            val-=step
    xs=[i for i in range(len(ys))
        ]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)


#end of functions

xs, ys =create_dataset(40,10,2, correlation=False)


# evaluating our parameters
m,b=best_fit_params(xs,ys)
print(f"slope is {m}\nintercept is {b}\n")

#forming the line with our parameters
regression_line=[m*x+b for x in xs]

#testing the line that was formed with original data
r_squared=coef_det(ys,regression_line)
print("rsquared is",r_squared*100)

flag=False
while(flag):
    X_input= float(input("Enter the number for which you want a prediciton: "))
    print(f"The predicted value is {m*X_input+b}")
    flag=bool(input("Continue?(0/1)"))  

X_predict=8
y_predict=m*X_predict+b


plt.scatter(xs,ys)
plt.scatter(X_predict,y_predict,color='green',s=100)
plt.plot(xs,regression_line)
plt.show()


