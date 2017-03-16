import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True`, we
#       return the second to last layer.
# SH notes:
# This essentially pulls AlexNet, "discards" the last layer.
# Gives us the net trained through layer 7.
# Now we add our own final (8th layer) to be trained on our data
# It basically sets all the weights and biases through layer fc7 from their pretrained network.
# Then we are going to "zero" the W and b for layer fc8.
#   Finally, we run our own fc on just this last layer.
# ?? Now that the network has been "fine-tuned" with our data,
# We Send the result through softmax to get probability mappings to traffic sign (signnames.csv) for our two test images
fc7 = AlexNet(resized, feature_extract=True)

# TODO: Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs. Assign the result of the softmax activation to `probs` below.
# HINT: Look at the final layer definition in alexnet.py to get an idea of
# what this should look like.
# use this shape for the weight matrix:
# SH
# second-to-last element of fc7 will be the size of the
# rem: shape == [x-dim, y-dim] (x-dim is 1, as its flat)
# get_shape..[-1] returns the length of layer7 (its flat),
#     which will be input size for layer8, and
#     nb_classes should be output size for layer 8
#     fc8W shape == (input_size, output_size) == (fc7_output_width, logits_size)
shape = (fc7.get_shape().as_list()[-1], nb_classes)

# SH
# set the variables AlexNet uses to hold our data (zeroed weights for layer8, our layer)
# we don't use AlexNet values for layer 8, we are going to train fc8W, fc8b with our data
fc8W = tf.Variable(tf.truncated_normal(shape))
fc8b = tf.Variable(tf.zeros(shape[len(shape)-1]))  # or shape(nb_classes)

# create new fc8 (logits) fully connected layer that is fed in:
#     - fc7 from pre-trained AlexNet nework,
#         (hence it's (feature trained) on lower levels)
#     - fc8W and fc8b, our "zeroed" initialized weights, for the last layer
#     so that this last, highest level, layer is trained ONLY with our data
#     (but used (feature training) of lower levels
# ???? train? (or just ?run????) output layer with our data
# TODO: WHY AREN'T WE TRAINING THE MODEL ON NEW DATA ??
# SO CONFUSED - how can it possible know how to classifify (get probs) these images ???
# Either we are RUNNING but NOT TRAINING, or I'm Seriously Confused about this process. Seems like some step is missing, that allows us to map logits over to our new dataset. Either Way, it seems that I'm seriously confused or missing something !!
logits = tf.matmul(fc7, fc8W) + fc8b   #fc8 layer, final output layer === logits

# return top 5 probabilities of what the image might be
probs = tf.nn.softmax(logits)
# AlexNet's runs its network using the variables as initialized above
#  our "extract call" fetched layer 1-7 weights and biases
#  then we set our own for layer 8, and defined what our layer 8 would be
#  and run softmax to determine what it thinks our images are
# SH end

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
