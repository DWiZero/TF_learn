import tensorflow as tf
from matplotlib import pylab
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('./MNIST_DATA', one_hot=True)
# print(mnist.train.images)
# 构造计算图
graph = tf.Graph()
with graph.as_default():
    W = tf.Variable(tf.random_normal([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 输入
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        pass
    # 前向推断过程
    with tf.name_scope("inference"):
        z = tf.matmul(x, W) + b
        pred = tf.nn.softmax(z)
        #
        # maxout = tf.reduce_max(z, axis=1, keep_dims=True)
        # W2 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.1))
        # b2 = tf.Variable(tf.zeros([1]))
        # pred = tf.nn.softmax(tf.matmul(maxout, W2) + b2)
        pass
    # 损失
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
        pass
    # 优化训练
    with tf.name_scope("train"):
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        pass
    # 模型评估
    with tf.name_scope("evaluate"):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pass

    tf.summary.FileWriter("./BP_model/",graph)

# 训练，保存模型
train_epochs = 25
bitch_size = 100
display_step = 1

# with tf.Session(graph=graph) as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epoch in range(train_epochs):
#         avg_cost = 0
#         total_batch = int(mnist.train.num_examples / bitch_size)
#         for i in range(total_batch):
#             bitch_xs, bitch_ys = mnist.train.next_batch(bitch_size)
#             _, c = sess.run([optimizer, cost], feed_dict={x: bitch_xs, y: bitch_ys})
#
#             avg_cost += c / total_batch
#         if (epoch + 1) % display_step == 0:
#             print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
#
#     print("Finished!")
#
#     save_path = tf.train.Saver().save(sess=sess, save_path="./BP_model/model")
#     _, train_accuracy = sess.run([optimizer, accuracy], feed_dict={x: mnist.train.images, y: mnist.train.labels})
#     print("Accuracy", train_accuracy)

    # correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # print("Accuracy",accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))

# 读取模型，验证
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    tf.train.Saver().restore(sess=sess, save_path="./BP_model/model")

    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    _, test_accuracy = sess.run([optimizer, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("Test_Accuracy", test_accuracy)


    xs,ys = mnist.train.next_batch(1)
    output = sess.run(pred,feed_dict={x:xs})
    pylab.imshow(xs.reshape(-1,28))
    pylab.show()
    print(output, ys)
    outputvalue = sess.run(tf.argmax(pred, 1),feed_dict={x:xs})
    ysvalue = sess.run(tf.argmax(ys, 1), feed_dict={x: xs})
    print(outputvalue, ysvalue)
