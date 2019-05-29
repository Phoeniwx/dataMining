import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

test_filepath="dota2Test.csv"
train_filepath="dota2Train.csv"
#y_all=np.loadtxt(test_filepath,delimiter=",",usecols=(0))
x_train=np.loadtxt(train_filepath,dtype=float,delimiter=",",usecols=np.arange(1,116))
y_train=np.loadtxt(train_filepath,dtype=float,delimiter=",",usecols=(0))
x_test=np.loadtxt(test_filepath,dtype=float,delimiter=",",usecols=np.arange(1,116))
y_test=np.loadtxt(test_filepath,dtype=float,delimiter=",",usecols=(0))

for i in range(y_train.size):
    if y_train[i] < 0:
        y_train[i]=0
for i in range(y_test.size):
    if y_test[i] < 0:
        y_test[i]=0

print(y_train)

model = Sequential()
model.add(Dense(115, input_dim=115, activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(115, activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20)
score = model.evaluate(x_test, y_test)
