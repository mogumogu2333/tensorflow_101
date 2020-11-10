import tensorflow as tf
from segment_inner_outer_range import segment_inner_range, segment_outer_range


def segment_lengths_to_sparse_tensor(values, segment_lengths):
    values, segment_lengths = map(tf.convert_to_tensor, [values, segment_lengths])
    max_len = tf.to_int64(tf.reduce_max(segment_lengths))
    batch_size = tf.to_int64(tf.size(segment_lengths))
    shape = tf.stack([batch_size, max_len], axis=0)
    segment_ids = segment_outer_range(segment_lengths)
    column_ids = segment_inner_range(segment_lengths)
    indices = tf.to_int64(tf.stack([segment_ids, column_ids], axis=1))
    return tf.SparseTensor(values=values, indices=indices, dense_shape=shape)

def sparse_tensor_to_dense(sp_tensor, default_value=None):
    # tensor_scatter_nd_update not supported; also scatter_update doesn't work
    # because it's applied to a variable, not a tensor
    if default_value in [0, None, False]:
        return tf.scatter_nd(indices=sp_tensor.indices,
        updates=sp_tensor.values, shape=sp_tensor.dense_shape)
    default_value = tf.cast(default_value, dtype=sp_tensor.dtype)
    mask = tf.scatter_nd(indices=sp_tensor.indices, updates=tf.fill(tf.shape(
        sp_tensor.values), True), shape=sp_tensor.dense_shape)
    other_indices = tf.where(tf.logical_not(mask))
    other_values = tf.fill(tf.shape(other_indices)[:1], default_value)
    indices = tf.concat([sp_tensor.indices, other_indices], axis=0)
    updates = tf.concat([sp_tensor.values, other_values], axis=0)
    return tf.scatter_nd(indices=indices, updates=updates, shape=sp_tensor.dense_shape)


def segment_tile(values, segment_lengths, segment_multiples, inner_tile=True):
    """For example
    outer_tile: [0, 1, 2, 3, 4], [2, 3], [2, 3] ->
    [0,1,0,1,2,3,4,2,3,4,2,3,4].
    inner_tile: [0, 1, 2, 3, 4], [2, 3], [2, 3] ->
    [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    """
    values, segment_lengths, segment_multiples = map(tf.convert_to_tensor,
        [values, segment_lengths, segment_multiples])
    sp_tensor = segment_lengths_to_sparse_tensor(values, segment_lengths)
    dense = sparse_tensor_to_dense(sp_tensor)
    mask_values = tf.ones_like(values, dtype=tf.bool)
    dense_mask = sparse_tensor_to_dense(segment_lengths_to_sparse_tensor(
        mask_values, segment_lengths))  # to undo sparse_tensor_to_dense
    max_multiple = tf.reduce_max(segment_multiples)
    mult_mask = tf.sequence_mask(segment_multiples)
    max_len = tf.reduce_max(segment_lengths)
    batch_size = tf.shape(segment_lengths)[0]
    if inner_tile:
        tile_fn = lambda x: tf.tile(tf.expand_dims(x, axis=2), [1, 1, max_multiple])
        # to undo max_multiple
        tile_mask = tf.tile(tf.expand_dims(mult_mask, axis=1), [1, max_len, 1])
    else:
        tile_fn = lambda x: tf.tile(x, [1, max_multiple])
        tile_mask = tf.reshape(tf.tile(tf.expand_dims(
            mult_mask, axis=2), [1, 1, max_len]), [batch_size, max_len * max_multiple])
    dense_tiled, mask_tiled = tile_fn(dense), tile_fn(dense_mask)
    final_mask = tf.logical_and(mask_tiled,  tile_mask)
    return tf.boolean_mask(dense_tiled, final_mask)

if __name__ == "__main__":
    with tf.Graph().as_default():
        query_ids = tf.constant([0,0,1,1,1])
        query_segment_ids = tf.constant([2,3])
        title_segment_ids = tf.constant([3,4])
        z = segment_tile(query_ids, query_segment_ids, title_segment_ids, inner_tile=True)
        
        sess = tf.Session()
        print(sess.run(z))
