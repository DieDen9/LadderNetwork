import tensorflow as tf
import input_data
import math
import os
import csv
from tqdm import tqdm
from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import confusion_matrix
import datetime


layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

L = len(layer_sizes) - 1  # number of layers

num_examples = 60000
num_epochs = 20
num_labeled = 100

starter_learning_rate = 0.02

decay_after = 15  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = int((num_examples/batch_size) * num_epochs ) # number of loop iterations

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
outputs = tf.placeholder(tf.float32)


def bi(inits, size, name):
    return tf.Variable(inits * tf.ones([size]), name=name)


def wi(shape, name):
    return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

shapes = list(zip(layer_sizes[:-1], layer_sizes[1:])) # shapes of linear layers
weights = {'W': [wi(s, "W") for s in shapes],  # Encoder weights
           'V': [wi(s[::-1], "V") for s in shapes],  # Decoder weights
           # batch normalization parameter to shift the normalized value
           'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(L)],
           # batch normalization parameter to scale the normalized value
           'gamma': [bi(1.0, layer_sizes[l+1], "beta") for l in range(L)]}

noise_std = 0.3  # scaling factor for noise used in corrupted encoder

# hyperparameters that denote the importance of each layer
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

join = lambda l, u: tf.concat([l, u], 0)
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))

training = tf.placeholder(tf.bool)

ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

# average mean and variance of all layers
running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]


def update_batch_normalization(batch, l):
    "batch normalize + update average mean and variance of layer l"
    mean, var = tf.nn.moments(batch, axes=[0])
    assign_mean = running_mean[l-1].assign(mean)
    assign_var = running_var[l-1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)


def encoder(inputs, noise_std):
    h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
    d = {}  # to store the pre-activation, activation, mean and variance for each layer
    # The data for labeled and unlabeled examples are stored separately
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
    for l in range(1, L+1):
        print("Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l])
        d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
        z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
        z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

        m, v = tf.nn.moments(z_pre_u, axes=[0])

        # if training:
        def training_batch_norm():
            # Training batch normalization
            # batch normalization for labeled and unlabeled examples is performed separately
            if noise_std > 0:
                # Corrupted encoder
                # batch normalization + noise
                z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
            else:
                # Clean encoder
                # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
            return z

        # else:
        def eval_batch_norm():
            # Evaluation batch normalization
            # obtain average mean and variance and use it to normalize the batch
            mean = ewma.average(running_mean[l-1])
            var = ewma.average(running_var[l-1])
            z = batch_normalization(z_pre, mean, var)
            # Instead of the above statement, the use of the following 2 statements containing a typo
            # consistently produces a 0.2% higher accuracy for unclear reasons.
            # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
            # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
            return z

        # perform batch normalization according to value of boolean "training" placeholder:
        z = tf.cond(training, training_batch_norm, eval_batch_norm)

        if l == L:
            # use softmax activation in output layer
            h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
            tf.summary.histogram("Encoder output",h)
        else:
            # use ReLU activation in hidden layers
            h = tf.nn.relu(z + weights["beta"][l-1])
            tf.summary.histogram("Encoder hidden layer",h)

        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
    d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
    return h, d

print("=== Corrupted Encoder ===")
y_c, corr = encoder(inputs, noise_std)

print("=== Clean Encoder ===")
y, clean = encoder(inputs, 0.0)  # 0.0 -> do not add noise
# print(y.shape)
print("=== Decoder ===")





def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    '''
Parameters:
    correct_labels                  : These are your true classification categories.
    predict_labels                  : These are you predicted classification categories
    labels                          : This is a lit of labels which will be used to display the axix labels
    title='Confusion matrix'        : Title for your matrix
    tensor_name = 'MyFigure/image'  : Name for the output summay tensor

Returns:
    summary: TensorFlow summary

Other items to note:
    - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
    - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels,labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(5, 5), dpi=240, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    # classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    # classes = ['\n'.join(wrap(l, 40)) for l in classes]
    classes = [str(i) for i in range(0,10)]
    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    ax.set_title(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary

def g_gauss(z_c, u, size):
    "gaussian denoising function proposed in the original paper"
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

    z_est = (z_c - mu) * v + mu
    return z_est

# Decoder
z_est = {}
d_cost = []  # to store the denoising cost of all layers
for l in range(L, -1, -1):
    print("Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l])
    z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
    m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
    if l == L:
        u = unlabeled(y_c)
    else:
        u = tf.matmul(z_est[l+1], weights['V'][l])
        # u = z_est[l+1]*2
    u = batch_normalization(u)

    z_est[l] = g_gauss(z_c, u, layer_sizes[l])
    z_est_bn = (z_est[l] - m) / v
    # append the cost of this layer to d_cost
    d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

# calculate total unsupervised cost by adding the denoising cost of all layers
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)
cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))  # supervised cost
loss = cost + u_cost  # total cost

pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))  # cost used for prediction

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1))  # no of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
tf.summary.scalar('accuracy', accuracy)
learning_rate = tf.Variable(starter_learning_rate, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print("===  Loading Data ===")
mnist = input_data.read_data_sets("MNIST_data", n_labeled=num_labeled, one_hot=True)

saver = tf.train.Saver()


print("===  Starting Session ===")
sess = tf.Session()

train_writer = tf.summary.FileWriter('./logs/', sess.graph)
merge = tf.summary.merge_all()
img_d_summary_writer = tf.summary.FileWriter('./logs/', sess.graph)

i_iter = 0

ckpt = tf.train.get_checkpoint_state('checkpoints/')  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    i_iter = (epoch_n+1) * (num_examples/batch_size)
    print("Restored Epoch ", epoch_n)
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    init = tf.global_variables_initializer()
    sess.run(init)

print("=== Training ===")
print("Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")

for i in tqdm(range(0, num_iter)): #i_iter
    images, labels = mnist.train.next_batch(batch_size)
    sess.run([train_step], feed_dict={inputs: images, outputs: labels, training: True})
    predictions = sess.run(y_N, feed_dict={inputs: images, training: True})

    if (i % 20) == 0:
        img_d_summary = plot_confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1), list(range(0,9)), tensor_name='dev/cm')
        img_d_summary_writer.add_summary(img_d_summary, i)
        summary , acc = sess.run([merge,accuracy], feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False})
        img_d_summary_writer.add_summary(summary, i)
        # print(sess.run(tf.argmax(y,1) , feed_dict={inputs: mnist.test.images}))

    if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
        epoch_n = i/(num_examples/batch_size)
        if (epoch_n+1) >= decay_after:
            # decay learning rate
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, save_path='/checkpoints/model.ckpt', global_step= tf.round(epoch_n)) #used tf.round() to avoid error
        print("Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%")
        with open('train_log', 'w') as train_log:
            # write test accuracy to file "train_log"
            train_log_w = csv.writer(train_log)
            log_i = [epoch_n] + sess.run([accuracy], feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False})
            train_log_w.writerow(log_i)

print("Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, training: False}), "%")

sess.close()