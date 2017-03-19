import pickle
import tensorflow as tf
from   sklearn.model_selection import train_test_split
from   sklearn.utils import shuffle
from   alexnet import AlexNet

# TODO: Load traffic signs data.

print("opening")
with open('train.p', 'rb') as f:
  data = pickle.load(f)
#see how our data is stored, so we can know how to extract  X and y from it
#print(data)
# ok, data is a dictionary, with 'features', 'coords, 'labels, 'sizes'
X = data['features']
y = data['labels']

print("data size X,y: ", len(X), len(y))
#temp for testing
#X = X[0:500]
#y = y[0:500]
#print("truncating to ", len(X), "data items")

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  #random_state is seed for random sampling

# TODO: Define placeholders and resize operation.
# get number of classes
def bufcount(filename):
    #http://stackoverflow.com/a/850962/5411817
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines
#num_classes = 43  ###?? # len of signnames.csv-2
num_classes = bufcount('./signnames.csv')
print(num_classes, "classes")

print(y.shape, "shape of y")

x_shape = (None,) + X_train.shape[1:]
#print("x_shape", x_shape)

x = tf.placeholder(tf.float32, x_shape)
y = tf.placeholder(tf.int32, None)
one_hot_y = tf.one_hot(y, num_classes)

mu = 0              #??
sigma = 0.1         #??

# Alexnet requires 227px x 227px images
resized = tf.image.resize_images(x, (227,227))
print("resized")

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
fc8_W  = tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma))
fc8_b  = tf.zeros(num_classes)
logits = tf.matmul(fc7, fc8_W) + fc8_b
logits = tf.nn.softmax(logits)
print("got logits")

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# training pipeline:
rate = 0.001        #??
EPOCHS = 10         #??
BATCH_SIZE = 128    #??

# temp testing
#EPOCHS = 4
#print("EPOCHS: ", EPOCHS)


# loss
cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
# training
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
# accuracy

# TODO: Train and evaluate the feature extraction model.

# define evaluation routines
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
  print("evaluating..")
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
    #temp testing
    #num_examples = 100
    print("training", num_examples, "examples" )

    print("Training...")
    print()

    for i in range(EPOCHS):
      X_train, y_train = shuffle(X_train, y_train)
      for offset in range (0, num_examples, BATCH_SIZE):
          print("     batch ", 1+offset//BATCH_SIZE, "of ", 1+ num_examples/BATCH_SIZE)
          end = offset + BATCH_SIZE
          batch_x = X_train[offset:end]
          batch_y = y_train[offset:end]
          sess.run(training_operation,
                   feed_dict={x: batch_x, y: batch_y})

        # evaluate the model
      validation_accuracy = evaluate(X_test, y_test)
      print("EPOCH {} ...".format(i+1))
      print("Validation Accuracy = {:.3f}".format(validation_accuracy))
      print()

    # save trained model
    saver.save(sess, './sh-trained')
    print("Model saved")


# evaluate model on test set
with tf.Session() as sess:
  saver.restore(sess, tf.train.latest_checkpoint('.'))

  test_accuracy = evaluate(X_test, y_test)
  print("Test Accuracy = {:.3f}".format(test_accuracy))

