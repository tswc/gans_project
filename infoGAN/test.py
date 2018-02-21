def conv2d(inputs, filters, kernel_size, strides, stddev=0.02, name="conv2d"):
    with tf.variable_scop(name):
        conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=4, strides=strides, 
                         kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                         bias_initializer=tf.constant_initializer(0.0),padding='same')
        return conv
		
		
def conv2d_transpose(inputs, filters, kernel_size, strides, stddev=0.02, name="conv2d_transpose"):
    with tf.variable_scop(name):
        conv_transpose = tf.layers.conv2d_transpose(
                         inputs=inputs, filters=filters, kernel_size=kernel_size,strides=strides, 
                         kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                         bias_initializer=tf.constant_initializer(0.0),padding= 'valid')
        return conv_transpose
		
def linear(inputs, output_dim, stddev=0.02, bias_start = 0.0):
    dense = tf.layers.dense(inputs=inputs, units=output_dim, 
                   kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                   bias_initializer=tf.constant_initializer(bias_start))
    return dense
	
	
	
def bn(x, is_training):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training)