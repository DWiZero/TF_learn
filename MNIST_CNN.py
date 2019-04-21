import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from matplotlib import pylab
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('./MNIST_DATA', one_hot=True)
# pylab.imshow(mnist.train.images[1].reshape(-1, 28))
# pylab.show()
# print(mnist.train.labels[1])
# 构造计算图
graph = tf.Graph()
with graph.as_default():
    # 输入
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])

        w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 64], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        w_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 10], stddev=0.1))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[10]))

        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)
        h_pool3 = tf.nn.avg_pool(h_conv3, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="SAME")

        h_pool3_flat = tf.reshape(h_pool3, [-1, 10])
        y_conv = tf.nn.softmax(h_pool3_flat)
        # 损失
    with tf.name_scope("loss"):
        # loss = -tf.reduce_mean(y * tf.log(y_conv))
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=1))
    with tf.name_scope("train"):
        optimizer = tf.train.AdadeltaOptimizer(0.15).minimize(loss)
        # optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.name_scope("evaluate"):
        corrent = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(corrent, "float"))
# 训练
train_epochs = 40
bitch_size = 100
display_step = 1

# with tf.Session(graph=graph) as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epoch in range(train_epochs):
#         avg_cost = 0
#         avg_accuracy = 0
#         total_batch = int(mnist.train.num_examples / bitch_size)
#         for i in range(total_batch):
#             bitch_xs, bitch_ys = mnist.train.next_batch(bitch_size)
#             _, c, train_accuracy = sess.run([optimizer, loss, accuracy], feed_dict={x: bitch_xs, y: bitch_ys})
#
#             avg_cost += c
#             avg_accuracy += train_accuracy
#         if (epoch + 1) % display_step == 0:
#             print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost / total_batch),
#                   "Train_accuracy %g" % (avg_accuracy / total_batch))
#
#     print("Finished!")
#
#     save_path = tf.train.Saver().save(sess=sess, save_path="./BP_CNN_model/model")

# 读取模型，验证
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    tf.train.Saver().restore(sess=sess, save_path="./BP_CNN_model/model")

    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # print("Accuracy", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    # _, test_accuracy = sess.run([optimizer, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    # print("Test_Accuracy", test_accuracy)

    output = tf.argmax(y_conv,1)
    # xs,ys = mnist.train.next_batch(1)
    ima = cv2.imread('./result/7.png')
    res = cv2.resize(ima, (28, 28), interpolation=cv2.INTER_CUBIC)
    xs = res[:,:,2]
    x_image = np.reshape(xs,[-1, 784])
    ys = 5
    outputvalue = sess.run(output,feed_dict={x:x_image})
    outputlist = sess.run(h_pool3_flat, feed_dict={x: x_image})
    # pylab.imshow(xs.reshape(-1,28))
    # pylab.show()
    print(outputvalue,ys)
    print(outputlist,ys)

