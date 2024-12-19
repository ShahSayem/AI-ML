# Slide page: 57
import numpy as np 
import tensorflow as tf

# depreciated modules can cause error
N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal((D, H)))
w2 = tf.Variable(tf.random_normal((H, D)))

h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
# diff = y_pred - y
# loss = tf.reduce_mean(tf.reduce_sum(diff*diff, axis=1))

# Use predefine common losses
loss = tf.losses.mean_squared_error(y_pred, y)

# Can use an "optimizer" to compute gradients and upadte weights
optimizer = tf.train.GradientDescentOptimizer(1e-5)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {x: np.random.randn(N, D),
              y: np.random.randn(N, D),}
    
    losses = []
    # Remember to execute the output of the optimizer!
    for t in range(50):               #######
        loss_val, _ = sess.run([loss, updates],
                               feed_dict = values)