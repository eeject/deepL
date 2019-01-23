#https://github.com/aditya9211/SVHN-CNN
from __future__ import absolute_import
from __future__ import print_function
import os
import time

from datetime import timedelta
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
plt.rcParams['figure.figsize'] = (16.0, 4.0) # Set default figure size

TENSORBOARD_SUMMARIES_DIR = '/tmp/svhn_classifier_logs'

# Open the file as readonly
h5f = h5py.File('SVHN_grey.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

# Close this file
h5f.close()

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)


def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    """ Plot nrows * ncols images from images and annotate the images
    """
    # Initialize the subplotgrid
    fig, axes = plt.subplots(nrows, ncols)

    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows * ncols)

    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat):

        # Predictions are not passed
        if cls_pred is None:
            title = "True: {0}".format(np.argmax(cls_true[i]))

        # When predictions are passed, display labels + predictions
        else:
            title = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), cls_pred[i])

            # Display the image
        ax.imshow(images[i, :, :, 0], cmap='binary')

        # Annotate the image
        ax.set_title(title)

        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])

def prepare_log_dir():
    '''Clears the log files then creates new directories to place
        the tensorbard log file.'''
    if tf.gfile.Exists(TENSORBOARD_SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(TENSORBOARD_SUMMARIES_DIR)
    tf.gfile.MakeDirs(TENSORBOARD_SUMMARIES_DIR)

def get_batch(X, y, batch_size=512):
    for i in np.arange(0, y.shape[0], batch_size):
        end = min(X.shape[0], i + batch_size)
        yield(X[i:end],y[i:end])

comp = 32*32
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
x = tf.placeholder(tf.float32, shape = [None, 32, 32, 1], name='Input_Data')
y = tf.placeholder(tf.float32, shape = [None, 10], name='Input_Labels')
y_cls = tf.argmax(y, 1)

discard_rate = tf.placeholder(tf.float32, name='Discard_rate')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def cnn_model_fn(features):
    """Model function for CNN."""

    # Input Layer
    input_layer = tf.reshape(features, [-1, 32, 32, 1], name='Reshaped_Input')

    # Convolutional Layer #1
    # with tf.name_scope('Conv1 Layer + ReLU'):

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # with tf.name_scope('Pool1 Layer'):
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    # with tf.name_scope('Conv2 Layer + ReLU'):
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # with tf.name_scope('Pool2 Layer'):
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=discard_rate)

    # Logits Layer
    # with tf.name_scope('Logits Layer'):
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits

max_epochs = 2
num_examples = X_train.shape[0]

prepare_log_dir()

# with tf.name_scope('Model Prediction'):
prediction = cnn_model_fn(x)
prediction_cls = tf.argmax(prediction, 1)
# with tf.name_scope('loss'):
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
    onehot_labels=y, logits=prediction))
# tf.summary.scalar('loss', loss)

# with tf.name_scope('Adam Optimizer'):
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Predicted class equals the true class of each image?
correct_prediction = tf.equal(prediction_cls, y_cls)

# Cast predictions to float and calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#merged_summary = tf.summary.merge_all()
sess = tf.Session()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

save_dir = 'checkpnts/'

# Create directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'svhn_single_greyscale')
#saver.restore(sess=session, save_path=save_path)
## No of example in each batch for updating weights
batch_size = 512

#Discarding or fuse % of neurons in Train mode
discard_per = 0.7
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())

## To calculate total time of training
train_loss = []
valid_loss = []
start_time = time.time()
for epoch in range(max_epochs):
    print('Training .........')
    epoch_loss = 0
    print()
    print('Epoch ', epoch + 1, ': ........ \n')
    step = 0

    ## Training epochs ....
    for (epoch_x, epoch_y) in get_batch(X_train, y_train, batch_size):
        _, train_accu, c = sess.run([optimizer, accuracy, loss],
                                    feed_dict={x: epoch_x, y: epoch_y, discard_rate: discard_per})
        train_loss.append(c)

        if (step % 40 == 0):
            print("Step:", step, ".....", "\nMini-Batch Loss   : ", c)
            print('Mini-Batch Accuracy :', train_accu * 100.0, '%')

            ## Validating prediction and summaries
            accu = 0.0
            for (epoch_x, epoch_y) in get_batch(X_val, y_val, 512):
                correct, _c = sess.run([correct_prediction, loss],
                                       feed_dict={x: epoch_x, y: epoch_y, discard_rate: 0.0})
                valid_loss.append(_c)
                accu += np.sum(correct[correct == True])
            print('Validation Accuracy :', accu * 100.0 / y_val.shape[0], '%')
            print()
        step = step + 1

    print('Epoch', epoch + 1, 'completed out of ', max_epochs)

## Calculate net time
time_diff = time.time() - start_time

## Testing prediction and summaries
accu = 0.0
for (epoch_x, epoch_y) in get_batch(X_test, y_test, 512):
    correct = sess.run([correct_prediction], feed_dict={x: epoch_x, y: epoch_y, discard_rate: 0.0})
    accu += np.sum(correct[correct == True])
print('Test Accuracy :', accu * 100.0 / y_test.shape[0], '%')
print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
print()

"""accu = 0.0
for (epoch_x , epoch_y) in get_batch(X_test, y_test, 512):
    correct = sess.run([correct_prediction], feed_dict={x: epoch_x, y: epoch_y, discard_rate: 0.0})
    predcls, classy = sess.run([prediction_cls, y_cls], feed_dict={x: epoch_x, y: epoch_y, discard_rate: 0.0})
    accumulate = np.sum((predcls == classy)*1)
    crct = np.sum(correct[correct == True])
    deaccumulate = np.sum((predcls != classy)*1)
    accu+= accumulate
    print('Test Accuracy :' , accumulate, crct, deaccumulate, accumulate+ deaccumulate, y_test.shape[0], '%')
print ()
"""
saver.save(sess=sess, save_path=save_path)

plot_images(X_train, 3, 6, y_train);
plt.show()
test_pred = []
for (epoch_x , epoch_y) in get_batch(X_test, y_test, 512):
    correct = sess.run([prediction_cls], feed_dict={x: epoch_x, y: epoch_y, discard_rate: 0.0})
    test_pred.append((np.asarray(correct, dtype=int)).T)

print ('Completed')
def flatten(lists):
    results = []
    for numbers in lists:
        for x in numbers:
            results.append(x)
    return np.asarray(results)

flat_array = flatten(test_pred)
flat_array = (flat_array.T)
flat_array = flat_array[0]

flat_array.shape

from sklearn.metrics import confusion_matrix

# Set the figure size
plt.figure(figsize=(12, 8))

# Calculate the confusion matrix
cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=flat_array)

# Normalize the confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True);
# Find the incorrectly classified examples
incorrect = flat_array != np.argmax(y_test, axis=1)

# Select the incorrectly classified examples
images = X_test[incorrect]
cls_true = y_test[incorrect]
cls_pred = flat_array[incorrect]

# Plot the mis-classified examples
plot_images(images, 3, 6, cls_true, cls_pred);
plt.show()

# Find the incorrectly classified examples
correct = np.invert(incorrect)

# Select the correctly classified examples
images = X_test[correct]
cls_true = y_test[correct]
cls_pred = flat_array[correct]

# Plot the mis-classified examples
plot_images(images, 3, 6, cls_true, cls_pred);
plt.show()
import matplotlib.pyplot as plt
plt.plot(train_loss ,'r')
plt.plot(valid_loss, 'g')
plt.show()