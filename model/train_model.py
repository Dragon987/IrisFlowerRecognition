import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, CuDNNLSTM
from keras.optimizers import Adam
import pydot
import graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import seaborn as sns
import pandas as pd
import numpy as np

dataset = sns.load_dataset("iris")

sns.set_style(style='ticks')
sns.set_palette('husl')
sns.pairplot(dataset.iloc[:, 0:6], hue='species')

x = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=0)


x_test = x_test.reshape([x_test.shape[0], 4])
x_train = x_train.reshape([x_train.shape[0], 4])
y_test = y_test.reshape([y_test.shape[0], 3])
y_train = y_train.reshape([y_train.shape[0], 3])

model = Sequential()

model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


""" Save model """
model_string = model.to_json()
with open("./model/model.json", 'w') as file:
    file.write(model_string)
model.save_weights('./model/model.h5')


# for i in range(30):
#     if np.argmax(y_test[i]) == 1:
#         print(x_test[i])

