import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow.examples.tutorials.mnist.input_data as mnist


data = mnist.read_data_sets('/mnist_pic/')
data_train = data.train.images
label_train = data.train.labels

# parameter
#num of examples      size of the image(784)
num_train, shape_img = data.train.images.shape

shape_img = data.train.images[0].shape[0]

# input of generater
shape_noise = 100

# units of h_g
g_units = 256

# units of h_d
d_units = 256

# learning rate
lr = 0.001


def generator(in_g, num_n, num_output, drop_rate=0.2, share=False):
    # hidden
    with tf.variable_scope("generator", reuse=share):
        h_g = tf.layers.dense(inputs=in_g, units=num_n)
        h_g = tf.nn.relu(h_g)
        h_g = tf.layers.dropout(h_g, drop_rate)

        # out
        logits_g = tf.layers.dense(h_g, num_output)
        ##???? tanh
        out_g = tf.tanh(logits_g)

    return logits_g, out_g


def discriminator(in_d, num_n, share=False, drop_rate=0.2):
    with tf.variable_scope("discriminator", reuse=share):
        # hidden
        h_d = tf.layers.dense(in_d, num_n)
        h_d = tf.nn.relu(h_d)
        h_d = tf.layers.dropout(h_d, drop_rate)

        # out
        logits_d = tf.layers.dense(h_d, 1)
        out_d = tf.sigmoid(logits_d)
    return logits_d, out_d

tf.reset_default_graph()

real_pic = tf.placeholder(tf.float32, [None, shape_img], name='in_discriminator')
noise_pic = tf.placeholder(tf.float32, [None, shape_noise], name='in_gererator')

# g
logits_g, out_g = generator(noise_pic, g_units, shape_img)

# d
logits_d_real, out_d_real = discriminator(real_pic, d_units)
logits_d_fake, out_d_fake = discriminator(out_g, d_units, share=True)

loss_d1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits_d_real,labels=tf.ones_like(logits_d_real)))

loss_d2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits_d_fake, labels=tf.zeros_like(logits_d_fake)))

# loss for d    d1+d2
loss_d = tf.add(loss_d1, loss_d2)

# loss for g
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logits_d_fake, labels=tf.ones_like(logits_d_fake)))

# parameter
parameter = tf.trainable_variables()
parameter_g = []
parameter_d = []
for para in parameter:
    if (para.name[0] == 'g'):
        parameter_g.append(para)
for para in parameter:
    if (para.name[0] == 'd'):
        parameter_d.append(para)

# define the optimizer of d and g

g_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_g, var_list=parameter_g)
d_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_d, var_list=parameter_d)

# batch
batch_size = 64

# epoch
epochs = 400

# num of samples
num_sample = 25


get_pic = []
losses = []

# save
saver = tf.train.Saver()
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for i in range(num_train//batch_size):
                batch = data.train.next_batch(batch_size)

                pic_batch = batch[0].reshape((batch_size, shape_img))
                pic_batch = 2*pic_batch-1

                # input for the generator
                in_g = np.random.uniform(-1, 1, size=(batch_size, shape_noise))

                # Optimzer
                _ = sess.run(d_optimizer, feed_dict={real_pic: pic_batch, noise_pic:in_g})
                _ = sess.run(g_optimizer, feed_dict={noise_pic: in_g})


            # loss
            train_loss_d = sess.run(loss_d, feed_dict={real_pic: pic_batch,
                                                       noise_pic: in_g})
            train_loss_d_real = sess.run(loss_d1, feed_dict={real_pic: pic_batch,
                                                       noise_pic: in_g})
            train_loss_d_fake = sess.run(loss_d2, feed_dict={real_pic: pic_batch,
                                                       noise_pic: in_g})

            train_loss_g = sess.run(loss_g, feed_dict={noise_pic: in_g})

            print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
              "Generator Loss: {:.4f}".format(train_loss_g))

            # save the losses
            losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

            # get the images in differen phases
            in_sample = np.random.uniform(-1, 1, size=(num_sample, shape_noise))
            pic_generate = sess.run(generator(noise_pic, g_units, shape_img,drop_rate=0.2, share=True),
                                    feed_dict={noise_pic: in_sample})
            get_pic.append(pic_generate)

            # save the checkpoints for NN transfer
            saver.save(sess, '/check_NN/para_NN.ckpt')

with open('samples.pkl', 'wb') as f:
    pickle.dump(get_pic, f)
