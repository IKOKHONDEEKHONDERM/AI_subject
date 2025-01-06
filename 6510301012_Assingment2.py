import numpy as np
import keras.api.models as mod
import keras.api.layers as lay
import matplotlib.pyplot as plt

model = mod.Sequential()
model.add(lay.SimpleRNN(units=1,input_shape=(1,1),activation="relu"))
model.summary()
model.save("RNN.h5")

pitch = 20
step = 20
N = 100
n_train = int(N*0.7)

def gen_data(x):
    return(x%pitch)/pitch

t = np.arange(1, N+1)
y = [gen_data(i) for i in t]
y = np.array(y)

plt.figure()
plt.plot(y)
plt.show()


#--------------------------------------------------------#
def convertToMatrix(data, step=1):
    X,Y = [],[]
    for i in range(len(data)-step):
        d = i + step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

train, test = y[0:n_train], y[n_train:N]

x_train, y_train = convertToMatrix(train,step)
x_test, y_test = convertToMatrix(test, step)

print("Dimension (Before): ", train.shape, test.shape)
print("Dimension (After) :", x_train.shape, x_test.shape)

model = mod.Sequential()
model.add(lay.SimpleRNN(units=32,input_shape=(step,1),activation='relu'))
model.add(lay.Dense(units=32))
model.compile(optimizer="adam", loss="mse",metrics=["accuracy"])
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

plt.plot(hist.history['loss'])
plt.show()
