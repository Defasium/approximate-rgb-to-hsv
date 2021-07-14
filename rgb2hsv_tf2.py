import tensorflow as tf
import numpy as np

def augment_tf(img, points=None):
    img = tf.cast(img, tf.float32)/255.
    batch_size = img.shape.as_list()[0]
    new_img = tf.map_fn(lambda x: tf.image.random_hue(x, 0.1), img)
    new_img = tf.map_fn(lambda x: tf.image.random_brightness(x, 0.2), new_img)
    new_img = tf.map_fn(lambda x: tf.image.random_saturation(x, 0.0, 1.2), new_img)
    new_img = tf.map_fn(lambda x: tf.image.random_contrast(x, 0.4, 1.2), new_img)
    new_img = tf.clip_by_value(new_img, 0.0, 1.0)
    return tf.cast(new_img*255, tf.uint8).numpy()

def augment_tf2(img, points=None):
    img = tf.cast(img, tf.float32)/255.
    batch_size = img.shape.as_list()[0]
    new_img = tf.image.rgb_to_hsv(img)
    new_img = tf.concat([new_img[:, :, :, :1]*0.2,
                         new_img[:, :, :, 1:2]*1.1,
                         new_img[:, :, :, 2:]*0.9], axis=-1)
    new_img = tf.image.hsv_to_rgb(new_img)
    new_img = tf.map_fn(lambda x: tf.image.random_contrast(x, 0.4, 1.2), new_img)
    new_img = tf.clip_by_value(new_img, 0.0, 1.0)
    return tf.cast(new_img*255, tf.uint8).numpy()
