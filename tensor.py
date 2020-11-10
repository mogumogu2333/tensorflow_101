import tensorflow as tf

def debug_tensor_shape():
    a = tf.constant([[1,2], [4,5]])
    print(a.get_shape())

if __name__ == "__main__":
    sess = tf.Session()
    # print(sess.run(debug_tensor_shape()))

