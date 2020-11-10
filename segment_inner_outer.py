import tensorflow as tf
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def segment_outer_range(segment_lengths, out_idx=tf.int32):
    """Given a list A of lengths, create [i for i, x in enumerate(A) for _ in range(x)]
    For example [2, 3, 1] -> [0, 0, 1, 1, 1, 2]
    """
    max_length = tf.reduce_max(segment_lengths)
    tiled_range = tf.tile(tf.expand_dims(tf.range(tf.size(segment_lengths, out_type=out_idx)), 1), [1, max_length])
    return tf.boolean_mask(
        tiled_range, tf.sequence_mask(segment_lengths, max_length))


def segment_inner_range(segment_lengths, out_idx=tf.int32):
    """Given a list A of lengths, create [i for x in A for i in range(x)].
    For example [2, 3, 1] -> [0, 1, 0, 1, 2, 0]
    """
    if segment_lengths.dtype != out_idx:
        segment_lengths = tf.cast(segment_lengths, out_idx)
    max_length = tf.reduce_max(segment_lengths)
    tiled_range = tf.tile(tf.expand_dims(tf.range(max_length), 0), [tf.size(segment_lengths), 1])
    return tf.boolean_mask(
        tiled_range, tf.sequence_mask(segment_lengths, max_length))


def segment_sum_with_batch_size(values, segment_ids, batch_size):
    dummy_val = '' if values.dtype == tf.string else 0
    augmented_ids = tf.concat([segment_ids, [batch_size]], axis=0)
    augmented_ids = tf.Print(augmented_ids, [augmented_ids], summarize=10, message="augmented_ids=")
    augmented_values = tf.concat([values, [dummy_val]], axis=0)
    augmented_values = tf.Print(augmented_values, [augmented_values], summarize=10, message="augmented_values=")
    sum = tf.segment_sum(augmented_values, augmented_ids)
    sum_1 = sum[:-1]
    return sum_1

def debug_segment_sum_with_batch_size():
    query_id_tiled = tf.constant([0,0,0,1,1,1,2,2,2,2,2,3,3,4,5,5])
    values = tf.ones_like(query_id_tiled)
    batch_size = tf.constant(8)
    return segment_sum_with_batch_size(values, query_id_tiled, batch_size)


def debug_stack():
    a = tf.constant([0,0,0,1,1,1,1,2,2,2,2,2])
    b = tf.constant([0,1,2,0,1,2,3,0,1,2,3,4])
    z = tf.stack([a, b],axis=1)
    sess = tf.Session()
        # print("Test tf.tile([1,2], [3]): ")
    print(a)
    print(b)
    print(sess.run(z))


def debug_tile():
    with tf.Graph().as_default():
        a = tf.constant([[0], [1], [2]],name='a') 
        a = tf.Print(a, [a], message="a=")
        b = tf.tile(a, [2,3], name="tile")
        sess = tf.Session()
        # print("Test tf.tile([1,2], [3]): ")
        # tf.print(a, output_stream=sys.stderr)
        print(b)
        print(sess.run(b))
        # [[0 0 0]
        # [1 1 1]
        # [2 2 2]
        # [0 0 0]
        # [1 1 1]
        # [2 2 2]]



def debug_reduce_sum():
    with tf.Graph().as_default():
        a = tf.constant([[2,3,1], [1,1,1]],name='a') 
        b = tf.reduce_sum(a, axis=0)
        sess = tf.Session()
        print("tf.reduce_sum(a): ")
        print(sess.run(b))

def debug_range():
    a = tf.constant(5)
    b = tf.range(a)
    sess = tf.Session()
    print(sess.run(b))

def debug_segment_outer_range():
    segment_lengths = tf.constant([3,4,5])
    max_length = tf.reduce_max(segment_lengths) # =5
    size = tf.size(segment_lengths, out_type=tf.int32) # =3
    r = tf.range(size) # = [0,1,2]
    e = tf.expand_dims(r, 1) # [[0], [1], [2]]
    tiled_range = tf.tile(e, [1, max_length])
#     [[0 0 0 0 0]
#     [1 1 1 1 1]
#     [2 2 2 2 2]]

    mask = tf.sequence_mask(segment_lengths, max_length)
    # [[ True  True  True False False]
#      [ True  True  True  True False]
#      [ True  True  True  True  True]]

    boolean_mask = tf.boolean_mask(tiled_range, mask)
    # [0,0,0,1,1,1,1,2,2,2,2,2]
    sess = tf.Session()
    print(sess.run(boolean_mask))

def debug_segment_inner_range():
    segment_lengths = tf.constant([3,4,5])
    size = tf.size(segment_lengths) # 3
    max_length = tf.reduce_max(segment_lengths) # =5
    r = tf.range(max_length) # = [0,1,2,3,4]
    
    e = tf.expand_dims(r, 0) # [[0 1 2 3 4]]
    tiled_range = tf.tile(e, [size, 1])
#     [[0 1 2 3 4]
#  [0 1 2 3 4]
#  [0 1 2 3 4]]

    mask = tf.sequence_mask(segment_lengths, max_length)
#     [[ True  True  True False False]
#  [ True  True  True  True False]
#  [ True  True  True  True  True]]
    boolean_mask = tf.boolean_mask(tiled_range, mask)
        # [0 1 2 0 1 2 3 0 1 2 3 4]

    sess = tf.Session()
    print(sess.run(boolean_mask))


if __name__ == "__main__":
    debug_tile()

    # sess = tf.Session()
    # print(sess.run(debug_segment_sum_with_batch_size()))
    # a = tf.constant([0,1,2])
    # b = tf.expand_dims(a, axis=1)
    # segment_len = tf.constant([2,2,2])
    # c = tf.tile(b,segment_len)
    # sess = tf.Session()
    # print(sess.run(c))

    