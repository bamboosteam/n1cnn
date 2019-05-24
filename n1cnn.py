# Import Libraries
import numpy as np
import tensorflow as tf
import sys as sy
import struct as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mymodule as mine

tf.logging.set_verbosity(tf.logging.INFO)

# Input data
target, width, height = mine.makeImage('/home2/bamboo/NeuralNetForPredict/Images/n1/ewpLH-ImgRes_0.dig')
low, width, height = mine.makeImage('/home2/bamboo/NeuralNetForPredict/Images/n1/ewpLH-LImgRes_0.dig')

# Define the size of extract block and the number of targets
numberOfTargets = 2
width_block, height_block = 6, 6
width_low, height_low = width_block, 1

# Make dataset
train_x, train_label, size_block = mine.makeDataset(target, low, width_block, height_block, width_low, height_low, numberOfTargets)

# Define the setting of layers
input_units, output_units = size_block, numberOfTargets
kernel_width, kernel_height = 3, 3
pool1_width, pool1_height = 2, 2
pooled1_width, pooled1_height = 2, 2
filter_amount = 16

# Define cnn model
def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 6, 6, 1])

    # Convolutional layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters = 32,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer 1

    # Convolutional layer 2 and Pooling layer 2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters = 64,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)


    # Dense layer
    conv2_flat = tf.reshape(conv2, [-1, 2*64*18])
    dense = tf.layers.dense(inputs=conv2_flat, units=64, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Create the Estimator
pixel_estimator = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/n1_CNN")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_x},
    y=train_label,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilities
pixel_estimator.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook])
