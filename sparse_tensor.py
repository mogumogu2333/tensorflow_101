import tensorflow as tf
from tensorflow.python.feature_column.utils import \
    sequence_length_from_sparse_tensor


def create_sparse_tensor():
    # 创建sparse tensor. 对应dense tensor的形状为batch_size * max_length. 
    # 想象成4个doc，第一个文档有3个token，第二个有2个token...
    batch_size=4
    max_length=3
    values = [0,0,0,1,2,2,3]
    indices = [[0,0],[0,1], [0,2], [1,0], [2,0], [2,1], [3,0]]
    dense_shape=[batch_size, max_length]
    return tf.SparseTensor(values=values, indices=indices, dense_shape=dense_shape)


def sparse_tensor_len():
    # sequence_length_from_sparse_tensor 获得每个文档的长度。
    # sparse tensor保存了batch size这么多个文档，由于sparse的格式，所以每个文档的长度是不一样的，sparse tensor的dense shape是最长的文档的长度。
    # 这个op计算每个文档的长度
    with tf.Graph().as_default():
        sess = tf.Session()
        sp_tensor = create_sparse_tensor()
        seq_len = sequence_length_from_sparse_tensor(sp_tensor)
        print(sess.run(seq_len))


if __name__ == "__main__":
    sparse_tensor_len()
    


