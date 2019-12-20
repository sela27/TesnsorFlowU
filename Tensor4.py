import tensorflow as tf
import numpy as np
import pandas as pd

def prep_data(train_siz=120, test_siz=30):

    cols = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid',
            'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label']

    iris_df = pd.read_csv('C:\\Users\\Owner\\PycharmProjects\\TesnsorFlowU\\voice.csv', header=None, names=cols)


    iris_df = iris_df.drop(iris_df.index[0])

    # Encode class
    class_name = ['male', 'female']
    iris_df['iclass'] = [class_name.index(class_str)
                         for class_str in iris_df['label'].values]

    # Random Shuffle before split to train/test
    orig = np.arange(len(iris_df))
    perm = np.copy(orig)
    np.random.shuffle(perm)
    iris = iris_df[
        ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid',
         'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'iclass']].values
    iris[orig, :] = iris[perm, :]

    # Split dataset
    trX = iris[:train_siz, :-1]
    teX = iris[train_siz:, :-1]
    trY = iris[:train_siz, -1]
    teY = iris[train_siz:, -1]

    return trX, trY, teX, teY



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prep_data(train_siz=2218, test_siz=950)

    with tf.Session() as sesh:
        y_train = sesh.run(tf.one_hot(y_train, 2))
        y_test = sesh.run(tf.one_hot(y_test, 2))

    learning_rate = 0.00001
    epochs = 4000
    batch_size = 100
    batches = int(x_train.shape[0] / batch_size)

    X = tf.placeholder(tf.float32, [None, 20])
    Y = tf.placeholder(tf.float32, [None, 2])

    W = tf.Variable(0.1*np.random.randn(20, 2).astype(np.float32))
    B = tf.Variable(0.1*np.random.randn(2).astype(np.float32))


    pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))
    #pred = tf.sigmoid(tf.add(tf.matmul(X, W), B))
    #cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred)))
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
                #print("W: " + str(W.eval(sesh)))
                #print("B: " + str(B.eval(sesh)))
                #print("x: " + str(x))
                #print("y: " + str(y))
                sesh.run(optimizer, feed_dict={X: x, Y: y})
                c = sesh.run(cost, feed_dict={X: x, Y: y})

            if not epoch % 500:
                print(f'epoch:{epoch:2d} cost={c:.16f}')

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc = accuracy.eval({X: x_test, Y: y_test})
        print(f'Accuracy: {acc * 100:.2f}%')
