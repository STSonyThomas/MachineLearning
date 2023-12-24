# Gradient Descent for Linear Regression
'''
basic formula for linear regression is y=wx+b
loss= mean sqaured error that is (expected output-recieved output)**2/ Number of samples
'''
# Initialise parameters
#Initialise gradient descent parameters
import numpy as np 
x = np.random.randn(10,1)
y = 2*x + np.random.rand() 
#Parameters
w= 0.0
b= 0.0
print(x.shape[0])

# Hyperparameter
learning_rate = 0.01
#create gradient descent function
def descend(x,y,w,b,learning_rate):
    dldw=0.0
    dldb=0.0
    N=x.shape[0]
    #loss=(y-(wx+b))**2
    for xi, yi in zip(x,y):
        dldw += -2*(yi-(w*xi+b))*(xi)
        dldb += -2*(yi-(w*xi+b))
    #Make an update to w parameter
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*(dldb)
    return w, b


#Iteratively make updates
# Loop through a bunch of epochs
for epoch in range(400):
    w, b = descend(x,y,w,b, learning_rate)
    yhat = w*x +b
    loss = np.divide(np.sum((y-yhat)**2, axis=0),x.shape[0])

    print(f"{epoch} loss is {loss}, parameters w:{w} b:{b}")
    #run gradient descent function
    pass