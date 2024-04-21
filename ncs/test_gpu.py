import tensorflow as tf

# Check TensorFlow for GPU usage
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define two constant matrices
a = tf.constant([[3, 3]])
b = tf.constant([[3],[3]])

# Run a matrix multiplication operation
c = tf.matmul(a, b)

print("Result of matrix multiplication: \n", c)

# Evaluate the computation graph
print("Session Run: ")
with tf.compat.v1.Session() as sess:
    result = sess.run(c)
    print(result)
