import time
import pickle
import tensorflow as tf
from   sklearn.model_selection import train_test_split
from   sklearn.utils import shuffle
from   alexnet import AlexNet

#set training values
mu = 0              #??
sigma = .01   #0.1  #??  #try .01 - solution uses .01
rate  = 0.001       #??
EPOCHS = 10         #??
# temp testing
EPOCHS = 4
print("EPOCHS: ", EPOCHS)
BATCH_SIZE = 128    #??

# read number of classes from signnames.csv
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
num_classes = bufcount('./signnames.csv') - 1
print(num_classes, "classes")
#num_classes = 43  # len of 'signnames.csv'

# TODO: Load traffic signs data.

print("opening")
with open('train.p', 'rb') as f:
  data = pickle.load(f)
#see how our data is stored, so we know how to extract  X and y from it
#print(data)
# findings: data is a dictionary: 'features', 'coords, 'labels, 'sizes'
X = data['features']
y = data['labels']

print("data size X, y: ", len(X), len(y))
#temp truncate data for initial testing
X = X[0:400]
y = y[0:400]
print("truncating to ", len(X), "data items")

# TODO: Split data into training and validation sets.
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.33, random_state=42)  #random_state is seed for random sampling

# TODO: Define placeholders and resize operation.
x_shape = (None,) + X_train.shape[1:]
#print("x_shape", x_shape)

features = tf.placeholder(tf.float32, x_shape)
labels   = tf.placeholder(tf.int64, None)
#one_hot_y = tf.one_hot(y, num_classes)

# Alexnet requires 227px x 227px images
resized = tf.image.resize_images(features, (227,227))
print("resized")

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], num_classes)
fc8_W  = tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma))
fc8_b  = tf.Variable(tf.zeros(num_classes))
logits = tf.matmul(fc7, fc8_W) + fc8_b  # or tf.nn.xw_plus_b(fc7, fc8W, fc8b)
# see notes below as to why not using softmax directly
#logits = tf.nn.softmax(logits)
print("got logits")

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# loss
#   http://stackoverflow.com/a/34243720/5411817
#   use this function instead of separate functions:
#   1) softmax with 2) cross_entropy and 3)(sparce) includes one-hot
#   softmax_cross_entropy_with_logits is more numerically stable/
#       accurate than running two steps of softmax, then cross_entropy
#   using the sparse_.. saves a step by not having to convert labels
#       to one-hot first
cross_entropy  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_operation = tf.reduce_mean(cross_entropy)

# training
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# accuracy
model_prediction = tf.argmax(logits, 1)
prediction_is_correct = tf.equal(model_prediction, labels)
accuracy_operation = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

# save trained model
saver = tf.train.Saver()


# TODO: Train and evaluate the feature extraction model.

# define evaluation routine
def evaluate(X_data, y_data):
  print("evaluating..")
  sess = tf.get_default_session()
  total_accuracy = 0
  total_loss = 0

  num_examples = X_data.shape[0]     #len(X_data)
  for offset in range(0, num_examples, BATCH_SIZE):
      X_batch = X_data[offset:offset+BATCH_SIZE]
      y_batch  = y_data[offset:offset+BATCH_SIZE]

      accuracy, loss = sess.run([accuracy_operation,
                                loss_operation],
                                feed_dict={features: X_batch,
                                           labels:   y_batch})
      batch_size = X_batch.shape[0]   #len(X_batch)
      total_accuracy += (accuracy * batch_size)
      total_loss     += (loss * batch_size)

  return total_accuracy/num_examples, total_loss/num_examples

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()

    num_examples = len(X_train) # == X_train.shape[0]
    #temp testing
    print(X_train.shape[0], "=?=", num_examples)
    print("training", num_examples, "examples, in", EPOCHS, "EPOCHS" )

    for i in range(EPOCHS):
      X_train, y_train = shuffle(X_train, y_train)
      t0 = time.time()
      for offset in range (0, num_examples, BATCH_SIZE):
          print("     batch ", 1+offset//BATCH_SIZE, "of ", 1 + num_examples/BATCH_SIZE, "batches,  on EPOCH", i+1, "of", EPOCHS, "EPOCHS")
          end = offset + BATCH_SIZE
          sess.run(training_operation,
                   feed_dict={features: X_train[offset:end],
                              labels:   y_train[offset:end]})

      # evaluate the model, print results
      # on validation set -- This is within training loop !
      validation_accuracy, validation_loss = evaluate(X_validate, y_validate)
      print("EPOCH {} ...".format(i+1))
      print("Time: {:.3f} minutes".format( float((time.time()-t0)/60)) )
      print("Validation Accuracy = {:.3f}".format(validation_accuracy))
      print("Validation Loss = {:.3f}".format(validation_loss))
      print()

    # save trained model
    saver.save(sess, './sh-trained')
    print("Model saved")
