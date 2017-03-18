import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
#   training datafile == "train.p"
#   ?solution? weights == bvlc-alexnet.py

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# TODO: Define placeholders and resize operation.
num_classes = ###?? # len of signnames.csv-2
x = tf.placeholder(float32, (None, ?,?,?))    #??
y = tf.placeholder(int32, None)
one_hot_y = tf.one_hot(y, num_classes)
#mu = 0                                       #??
#sigma =                                      #??
resized = tf.image.resize(x, (227,227)) # Alexnet requires 227px x 227px images

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], num_classes)
fc8_W  = tf.Variable(truncated_normal(shape))#, mean=mu, stddev=sigma)
fc8_b  = tf.zeros(num_classes)
logits = tf.matmul(fc7, fc8_W) + fc8_b
logits = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# training pipeline:
rate = 0.001        #??
EPOCHS = 10         #??
BATCH_SIZE = 128    #??
#mu = 0                                       #??
#sigma =                                      #??


# loss
cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
# accuracy
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
# training
training_operation = optimizer.minimize(loss_operation)


# TODO: Train and evaluate the feature extraction model.

# evaluation funcs
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset:offset+BATCH_SIZE]
        batch_y = y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation,
                            feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x = X_train[offset:end]
            batch_y = y_train[offset:end]
            sess.run(training_operation,
                     feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    #saver.save(sess, './sh-train-feature-extract-alexnet')
    print("Model saved")


# evaluate model on test set
with tf.Session() as sess:
    #saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

