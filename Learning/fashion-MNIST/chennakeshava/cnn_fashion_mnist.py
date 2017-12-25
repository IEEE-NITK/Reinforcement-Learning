from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#print(dir(__name__))
# help('__main__')

#print(inspect.getmembers(__name__))


def cnn_model(features, labels, mode):
	"""Model function for cnn prediction"""

	input_layers = tf.reshape(features["x"],[-1, 28, 28, 1]) # batch_size, image_width, image_height, channel_size
	# batch_size will be calculated dynamically - it's a hyper parameter that has to be tuned
	conv1 = tf.layers.conv2d(inputs=input_layers, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu) #same?
	# output has dimensions of 28 by 28 by 32
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
	# output has dimensions of 14 by 14 by 32

	conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu) #same?
	# output has dimensions of 14 by 14 by 64

	pool2 = tf.layers.max_pooling2d(inputs=conv2, strides=2, pool_size=2)
	# output has dimensions of 7 by 7 by 64

	# pool layers cut down the spatial dimensions by half, but they preserve the depth of the input

	# to pass pool2 to FC layers, convert it to the form [batch_size, features]
	pool2_flat = tf.reshape(pool2, [-1, 64 * 7 * 7])


	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

	#applying dropout to FC layer to prevent overfitting

	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN) # 40% of units are switched off
	# dropout is applied only during training phase, that's what the training attribute specifies

	logits = tf.layers.dense(inputs=dropout, units=10) # logits is of dimensions [batch_size, 10] ==> gives raw predictions

	predic = {"classes" : tf.argmax(input=logits,axis=1), "probs" : tf.nn.softmax(logits, name="softmax_tensor")} 
	# setting up the name explicitly, so that it's useful for logging purposes
	# why axis = 1 ???

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predic)

	# using cross-entropy as the loss function
	# but cross-entropy needs one-hot encoding of data

	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

	loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=onehot_labels)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

		return tf.estimator.EstimatorSpec(loss=loss, train_op=train_op, mode=mode)

	eval_metric_ops	= {
		"accuracy" : tf.metrics.accuracy(labels=labels, predictions=predic["classes"])
	}

	return tf.estimator.EstimatorSpec(eval_metric_ops=eval_metric_ops, mode=mode, loss=loss)

def main(unused_argv):

	mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot = False, validation_size = 0)

	train_data = mnist.train.images # returns an np array ?
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="./")

	tensors_to_log = {"probabilities" : "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": train_data},
	    y=train_labels,
	    batch_size=200,
	    num_epochs=None,
	    shuffle=True)
	

	mnist_classifier.train(
	    input_fn=train_input_fn,
	    steps=2000,
	    hooks=[logging_hook])

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": eval_data},
	    y=eval_labels,
	    num_epochs=1,
	    shuffle=False)

	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

	print(eval_results)

if __name__ == "__main__":
  tf.app.run()


