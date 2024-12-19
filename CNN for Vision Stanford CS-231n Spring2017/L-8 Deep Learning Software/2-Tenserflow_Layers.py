# Slide page: 59
import numpy as np 
import tensorflow as tf

# depreciated modules can cause error
N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

# Used Xavier initializer
init = tf.contrib.layers.xavier_initializer()
# tf.layers automatically sets up weight (and bias) for us!
h = tf.layers.dense(inputs = x, units = H, activation = tf.nn.relu, kernel_initializer = init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_ninitializer = init)

# Use predefine common losses
loss = tf.losses.mean_squared_error(y_pred, y)

# Can use an "optimizer" to compute gradients and upadte weights
optimizer = tf.train.GradientDescentOptimizer(1e0)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {x: np.random.randn(N, D),
              y: np.random.randn(N, D),}
    
    # Remember to execute the output of the optimizer!
    for t in range(50):               #######
        loss_val, _ = sess.run([loss, updates],
                               feed_dict = values)