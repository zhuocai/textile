import tensorflow as tf
import tensorflow.keras.backend as K


def ssim_metric(dynamic_range):
    def ssim(imgs_true, imgs_pred):
        return K.mean(tf.image.ssim(imgs_true, imgs_pred, dynamic_range), axis=-1)

    return ssim


def mssim_metric(dynamic_range):
    def mssim(imgs_true, imgs_pred):
        return K.mean(
            tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range), axis=-1
        )

    return mssim


def cai_l2_metric():
    def l2_metric(imgs_true, output):
        x1, x2 = output
        return tf.math.reduce_sum(tf.square(x1 - x2))
    return l2_metric
