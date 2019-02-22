import tensorflow as tf
a = tf.constant([2.5,2])
b = tf.constant([3.6], dtype=tf.float32)
total=tf.add(a,b)
sess =tf.Session()
sess =tf.InteractiveSession()
print(sess.run(total))
