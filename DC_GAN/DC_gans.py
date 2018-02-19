import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow.examples.tutorials.mnist.input_data as mnist
from TJ_package.gans_api import *
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Get data from mnist
data = mnist.read_data_sets('/mnist_pic/')
data_train = data.train.images
label_train = data.train.labels
num_train, shape_img = data.train.images.shape
shape_img = [-1,28,28,1]

# batch
batch_size = 64

# epoch
epochs = 5

# num of samples
num_sample = 25

shape_noise = 100
get_pic = []
losses = []
steps = 0
tf.reset_default_graph()
# inputs
real_pic, noise_pic = set_input(shape_noise, [-1, 28, 28, 1])
loss_g, loss_d = loss(real_pic, noise_pic, 1)
g_optimizer, d_optimizer = optimize(loss_g, loss_d)

# save
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for i in range(num_train // batch_size):
            steps +=1
            batch = data.train.next_batch(batch_size)

            pic_batch = batch[0].reshape((batch_size, shape_img[1],
                                          shape_img[2], shape_img[3]))

            pic_batch = 2 * pic_batch - 1

            # input for the generator
            in_g = np.random.uniform(-1, 1, size=(batch_size, shape_noise))

            # Optimzer
            _ = sess.run(d_optimizer, feed_dict={real_pic: pic_batch, noise_pic: in_g})
            _ = sess.run(g_optimizer, feed_dict={noise_pic: in_g})

        # loss
        train_loss_d = sess.run(loss_d, feed_dict={real_pic: pic_batch,
                                                   noise_pic: in_g})
        #         train_loss_d_real = sess.run(loss_d1, feed_dict={real_pic: pic_batch,
        #                                                    noise_pic: in_g})
        #         train_loss_d_fake = sess.run(loss_d2, feed_dict={real_pic: pic_batch,
        #                                                    noise_pic: in_g})

        train_loss_g = sess.run(loss_g, feed_dict={real_pic: pic_batch,
                                                   noise_pic: in_g})

        print("Epoch {}/{}....".format(e + 1, epochs),
              "Discriminator Loss: {:.4f}....".format(train_loss_d),
              "Generator Loss: {:.4f}....".format(train_loss_g))

        # save the losses
        # if steps % 101 == 0:
        #     print("Epoch {}/{}....".format(e + 1, epochs),
        #       "Discriminator Loss: {:.4f}....".format(train_loss_d),
        #       "Generator Loss: {:.4f}....".format(train_loss_g))
        losses.append((train_loss_d, train_loss_g))

        # get the images in differen phases
        in_sample = np.random.uniform(-1, 1, size=(num_sample, shape_noise))
        pic_generate = sess.run(generator(noise_pic,  shape_img[-1], share=True),
                                feed_dict={noise_pic: in_sample})
        get_pic.append(pic_generate)
        # save the checkpoints for NN transfer
        saver.save(sess, '/check_NN/para_NN.ckpt')


# with open('samples.pkl', 'wb') as f:

#     pickle.dump(get_pic, f)
