import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def test_l2_norm():
    a = tf.constant([[1,1,1,1], [2,2,2,2], [3,3,3,3]], dtype=tf.float32)
    b = tf.constant([[5,5,5,5], [6,6,6,6], [7,7,7,7]], dtype=tf.float32)
    normalize_a = tf.nn.l2_normalize(a,1)        
    normalize_b = tf.nn.l2_normalize(b,1)
    # cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
    cos_similarity = tf.reduce_sum(normalize_a * normalize_b, axis=1)
    return normalize_a, normalize_b,cos_similarity


def test_cosine_sim():
    a = tf.constant([1.1,1.1,1.1],  dtype=tf.float32)
    b = tf.constant([5.5,5.5,5.5], dtype=tf.float32)
    normalize_a = tf.nn.l2_normalize(a,1)        
    normalize_b = tf.nn.l2_normalize(b,1)
    cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
    return normalize_a, normalize_b, cos_similarity
    

def test_reduce_sum():
    a = tf.constant([[[1,2],[3,4]],[[1,2],[3,4]]])
    b = tf.reduce_sum(a, axis=[1,2])
    print(b)
    return b


def sq_sum(alist):
    return sum([x**2 for x in alist])

if __name__ == "__main__":
    # a = tf.constant([1,2])
    # b = tf.constant([3,6])
    # c = a*b
    sess = tf.Session()
    # a, b, sim = test_l2_norm()
    # alist = sess.run(a).tolist()
    # print(sess.run(a))
    # print(sess.run(b))
    # print(sess.run(sim))
    # print(sq_sum(alist))
    b = test_reduce_sum()
    print(sess.run(b))

