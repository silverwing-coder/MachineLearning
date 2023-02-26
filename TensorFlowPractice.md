<h4>Update:  Feb. 2023</h4>

<h2>TensorFlow Practice</h2>

<em><h4>Steps in modeling with TensorFlow </h4></em>

1. Get data ready
2. Creating a model - define the input and output layers, hiddden layers of a deep learning model <br/>

```py
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1) # number of hidden layers
])
```

3. Compiling a model - define the loss function (how wrong it is) and the optimizer (imporve the patterens), and evaluation matrix <br/>
    - Loss: how wrong your model's predictions are compared to the truth levels (minimize)
    - Optimizer: how your model should update its internal patterns to a better predictions
    - Metrics: human intrepretable values for how well your model is doing

```py
model.compile(loss=tf.keras.lossess.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])
```

4. Fitting a model - letting the model try to find patterns between x & y <br/>
    - Epochs: how many times the training model will go through all of the training examples

```py
model.fit(x_train, y_train, epochs=5)
```

5. Evaluate the model<br/>

```py
model.evaluate(x_test, y_test)
```

6. Improve through experimentation<br/>
    - Create model: add more layers, increase the number of hidden units (neurons), change activation function in eacy layer

```py
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1),
])
```

    - Compiling model: change the optimization function or learning rate of the optimization function

```py
model.compile(loss=tf.keras.lossess.mae,
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=["mae"])
```

    - Fitting model: more epochs, or more data

```py
model.fit(x_train_full, y_train_full, epochs=100)
```

7. Save and reload your trained model<br/>
 </li>
