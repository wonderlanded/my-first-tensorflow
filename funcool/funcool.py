import tensorflow as tf
import pandas as pd
import os

file_path = 'funcool.csv'

data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), file_path))

fun = data[['fun']]
cool = data[['cool']]
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(fun, cool, epochs=1000)

res = model.predict([50])

print(res)