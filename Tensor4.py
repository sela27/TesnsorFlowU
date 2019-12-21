import tensorflow as tf
import numpy as np
import pandas as pd

def prep_data(train_siz, test_siz):

    cols = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid',
            'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label']

    df = pd.read_csv('C:\\Users\\Owner\\PycharmProjects\\TesnsorFlowU\\voice.csv', header=None, names=cols)


    df = df.drop(df.index[0])

    # Encode class
    class_name = ['male', 'female']
    df['label_num'] = [class_name.index(class_str)
                         for class_str in df['label'].values]

    # Random Shuffle before split to train/test
    orig = np.arange(len(df))
    perm = np.copy(orig)
    np.random.shuffle(perm)
    data = df[
        ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid',
         'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label_num']].values
    data[orig, :] = data[perm, :]

    # Split dataset
    trX = data[:train_siz, :-1]
    teX = data[train_siz:, :-1]
    trY = data[:train_siz, -1]
    teY = data[train_siz:, -1]

    return trX, trY, teX, teY



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prep_data(train_siz=2218, test_siz=950)

    with tf.Session() as sesh:
        y_train = sesh.run(tf.one_hot(y_train, 2))
        y_test = sesh.run(tf.one_hot(y_test, 2))

    learning_rate = 0.00002
    epochs = 200000
    batch_size = 1500
    batches = int(x_train.shape[0] / batch_size)

    X = tf.placeholder(tf.float32, [None, 20])
    Y = tf.placeholder(tf.float32, [None, 2])

    W = tf.Variable(0.001*np.random.randn(20, 2).astype(np.float32))
    B = tf.Variable(0.001*np.random.randn(2).astype(np.float32))


    pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0))))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True).minimize(cost)

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        c = 0
        op = 0

        for epoch in range(epochs):
            for i in range(batches):
                offset = i * epoch
                x = x_train[offset: offset + batch_size]
                y = y_train[offset: offset + batch_size]
                sesh.run(optimizer, feed_dict={X: x, Y: y})
                c = sesh.run(cost, feed_dict={X: x, Y: y})

            if not epoch % 500:
                print(f'epoch:{epoch:2d} cost={c:.16f}')

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc = accuracy.eval({X: x_test, Y: y_test})
        conf_mat = tf.math.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(pred, 1),num_classes=2)
        conf_mat_to_print = sesh.run(conf_mat, feed_dict={X: x_test, Y: y_test})
        print(f'Accuracy: {acc * 100:.2f}%')
        print(conf_mat_to_print)
