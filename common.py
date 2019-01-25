import tensorflow as tf
import json
import datasets
import datetime


def name(args, prefix=""):
    if "logdir" not in args:
        name = "{}-{}-bs{}-apw{}-fw{}-ctx{}-nr{}-sr{}".format(
            prefix,
            datetime.datetime.now().strftime("%m-%d_%H%M%S"),
            args["batch_size"],
            args["annotations_per_window"],
            args["frame_width"],
            args["context_width"],
            args["note_range"],
            args["samplerate"],
        )
        args["logdir"] = "models/" + name

        print()
        print(name)
        print()


def bn_conv(inputs, filters, size, strides, padding, activation=None, dilation_rate=1, training=False):
    name = "bn_conv{}-f{}-s{}-dil{}-{}".format(size, filters, strides, dilation_rate, padding)
    with tf.name_scope(name):
        l = tf.layers.conv1d(inputs, filters, size, strides, padding, activation=None, use_bias=False, dilation_rate=dilation_rate)
        l = tf.layers.batch_normalization(l, training=training)
        if activation:
            return activation(l)
        else:
            return l


def conv(inputs, filters, size, strides, padding, activation=None, dilation_rate=1, training=False):
    name = "conv{}-f{}-s{}-dil{}-{}".format(size, filters, strides, dilation_rate, padding)
    return tf.layers.conv1d(inputs, filters, size, strides, padding, activation=activation, dilation_rate=dilation_rate, name=name)
