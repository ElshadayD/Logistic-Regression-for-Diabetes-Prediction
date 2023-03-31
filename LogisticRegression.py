import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return  1 / (1 + np.exp(-z))

def compute_cost(prediction, y):
    return (-y * np.log(prediction) - (1 - y) * np.log(1 - prediction)).mean()

def compute_gradient(X, prediction, y):
    d_dw=np.dot(X.T, (prediction- y)) / y.shape[0]
    d_db=np.sum(prediction- y) / y.shape[0]
    return d_dw,d_db

def gradient_descent(X, y, w,b, alpha, num_iters):
    cost_array = np.zeros(num_iters)
    for i in range(num_iters):
        prediction = sigmoid(np.dot(X, w)+b)
        cost= compute_cost(prediction, y)
        cost_array[i] = cost
        d_dw,d_db= compute_gradient(X, prediction, y)
        w = w - (alpha * d_dw)
        b = b - (alpha * d_db)
    return w,b, cost_array

def plotGraph(iterations, cost):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost, 'b')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs Iterations')
    plt.show()

def main():

    data = pd.read_csv('diabetes.csv')

    X = data[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
    y = data['Outcome']


    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    w= np.zeros(X.shape[1])
    b=0
    alpha = 0.0001
    iterations = 10000
    pediction = sigmoid(np.dot(X, w)+b)
     print("Initial cost value for w and b values {0} and {1}is: {2}".format(w, b,compute_cost(prediction,y)))

    w, b,cost_num = gradient_descent(X, y, w,b, alpha, iterations)

    prediction= sigmoid(np.dot(X, w))
    print("Final cost value for w and b values {0} and {1}is: {2}".format(w, b,compute_cost(prediction,y)))
    plotGraph(iterations, cost_num)

if __name__ == "__main__":
    main()
