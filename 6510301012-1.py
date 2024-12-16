from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
import numpy as np

x1,y1 = make_blobs(n_samples=100,
                   n_features=2,
                   centers=1,
                   center_box=(2.0,2.0),
                   cluster_std=0.25,
                   random_state=69)
x2,y2 = make_blobs(n_samples=100,
                   n_features=2,
                   centers=1,
                   center_box=(3.0,3.0),
                   cluster_std=0.25,
                   random_state=69)

def decision_function(x1,x2):
    return x1 + x2 - 0.5

x1_range = np.linspace(-1,2,500)
x2_range = np.linspace(-1,2,500)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
g_values = decision_function(x1_grid,x2_grid)

X = np.vstack((x1, x2))
y = np.hstack((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))  # Class 1 = 0, Class 2 = 1
X = np.c_[X, np.ones(X.shape[0])]# add bias term in x

# Activation function (Step Function)
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron Learning Algorithm
def perceptron(X, y, learning_rate=0.1, epochs=100):
    weights = np.zeros(X.shape[1])  # น้ำหนักเริ่มต้นเป็นศูนย์
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            prediction = step_function(np.dot(X[i], weights))  # คำนวณผลลัพธ์
            error = y[i] - prediction  # คำนวณข้อผิดพลาด
            weights += learning_rate * error * X[i]  # ปรับปรุงน้ำหนัก
    return weights

# trian Perceptron
weights = perceptron(X, y)

def plot_decision_boundary(X, y, weights):
    plt.figure()
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 2')
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(weights[0] * x_vals + weights[2]) / weights[1]  # Decision Boundary
    plt.plot(x_vals, y_vals, color='black', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()

plot_decision_boundary(X, y, weights)

fig = plt.figure()
fig.suptitle("Data Sample")
plt.scatter(x1[:,0], x1[:,1], c='red',linewidths=1, alpha=0.6, label="Class 1")
plt.scatter(x2[:,0], x2[:,1], c='blue',linewidths=1, alpha=0.6, label="Class 2")
plt.xlabel('Feature 1', fontsize = 10)
plt.xlabel('Feature 2', fontsize = 10)
plt.grid(True, axis='both')
plt.legend(loc='lower right')
plt.show()
fig.savefig('Out1 - Data Sample.png')

plt.figure()
plt.contourf(x1_grid, x2_grid, g_values, levels = [-np.inf,0,np.inf], colors = ['red','blue'],alpha = 0.5)
plt.contour(x1_grid, x2_grid, g_values, levels =[0], color = 'black',linewidths= 2 )
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.grid('Decision Plane')
plt.grid(True)
plt.show()