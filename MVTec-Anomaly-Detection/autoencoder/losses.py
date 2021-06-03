import tensorflow as tf


def ssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        # return (1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)) / 2

        return 1 - tf.image.ssim(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, dynamic_range))

    return loss


def mssim_loss(dynamic_range):
    def loss(imgs_true, imgs_pred):
        return 1 - tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)

        # return 1 - tf.reduce_mean(
        #     tf.image.ssim_multiscale(imgs_true, imgs_pred, dynamic_range)
        # )

    return loss


def l2_loss(imgs_true, imgs_pred):
    # return 2 * tf.nn.l2_loss(imgs_true - imgs_pred)
    return tf.nn.l2_loss(imgs_pred)


def cai_l2_loss(imgs_true, output):
    # x1, x2 = output
    return tf.nn.l2_loss(output)


def correlation(x, y):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


def cai_cor_loss(imgs_true, output):
    x1, x2 = output
    return correlation(x1, x2)
