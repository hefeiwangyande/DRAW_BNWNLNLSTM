from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

def batch_norm(x,
               n_out,
               is_training,
               reuse=None,
               gamma=None,
               beta=None,
               axes=[0, 1, 2],
               eps=1e-3,
               scope="bn",
               name="bn_out",
               return_mean=False):
  with tf.variable_scope(scope, reuse=reuse):
    emean = tf.get_variable(name="ema_mean", shape=[n_out], trainable=False)
    evar = tf.get_variable(name="ema_var", shape=[n_out], trainable=False)
    if is_training:
      batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
      batch_mean.set_shape([n_out])
      batch_var.set_shape([n_out])
      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      ema_apply_op_local = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op_local]):
        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
      emean_val = ema.average(batch_mean)
      evar_val = ema.average(batch_var)
      with tf.control_dependencies(
          [tf.assign(emean, emean_val), tf.assign(evar, evar_val)]):
        normed = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, eps, name=name)
    else:
      normed = tf.nn.batch_normalization(
          x, emean, evar, beta, gamma, eps, name=name)
  if return_mean:
    if is_training:
      return normed, mean
    else:
      return normed, emean
  else:
    return normed

def batch_renorm(x,
               n_out,
               is_training,
               reuse=None,
               gamma=None,
               beta=None,
               axes=[0, 1, 2],
               eps=1e-3,
               scope="rebn",
               name="rebn_out",
               return_mean=False):
  with tf.variable_scope(scope,reuse=reuse):
    RMAX = 3
    DMAX = 5
    emean = tf.get_variable("ema_mean", [n_out], trainable=False)
    evar = tf.get_variable("ema_var", [n_out], trainable=False)
    if is_training:
      batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
      batch_mean.set_shape([n_out])
      batch_var.set_shape([n_out])

      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      # if reuse:
      #   with tf.variable_scope(tf.get_variable_scope(), reuse=None):
      #     ema_apply_op_local = ema.apply([batch_mean, batch_var])
      # else:
      ema_apply_op_local = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op_local]):
          mean, var = tf.identity(batch_mean), tf.identity(batch_var)
      emean_val = ema.average(batch_mean)
      evar_val = ema.average(batch_var)

      moving_inv = tf.rsqrt(evar_val +eps)
      r = tf.clip_by_value(tf.sqrt(batch_var + eps) * moving_inv,
                           1 / RMAX,
                           RMAX)
      d = tf.clip_by_value((batch_mean - emean_val) * moving_inv,
                           -DMAX,
                           DMAX)
      scale = tf.stop_gradient(r, name='renorm_r')
      offset = tf.stop_gradient(d, name='renorm_d')
      if gamma is not None:
        scale *= gamma
        offset *= gamma
      if beta is not None:
        offset += beta
      with tf.control_dependencies(
          [tf.assign(emean, emean_val), tf.assign(evar, evar_val)]):
        normed = tf.nn.batch_normalization(
            x, mean, var, offset, scale, eps, name=name)
    else:
      normed = tf.nn.batch_normalization(
          x, emean, evar, beta, gamma, eps, name=name)
  if return_mean:
    if is_training:
      return normed, mean
    else:
      return normed, batch_mean
  else:
    return normed

  # tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None,
  #                               epsilon=0.00001, scale=True, scope="d_h1_conv")

#
#
# def batch_renorm(x,
#                n_out,
#                is_training,
#                reuse=None,
#                gamma=None,
#                beta=None,
#                axes=[0, 1, 2],
#                eps=1e-3,
#                scope="rebn",
#                name="rebn_out",
#                return_mean=False):
#   with tf.variable_scope(scope, reuse=reuse):
#     RMAX = 3
#     DMAX = 5
#     decay=0.9
#
#     pop_mean = tf.get_variable('pop_mean', [n_out], initializer=tf.zeros_initializer, trainable=False)
#     pop_var = tf.get_variable('pop_var', [n_out], initializer=tf.ones_initializer, trainable=False)
#     batch_mean, batch_var = tf.nn.moments(x, axes)
#
#     pop_mean = batch_mean
#     pop_var=batch_var
#
#     train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
#     train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
#
#
#
#     moving_inv = tf.rsqrt(pop_var + eps)
#     r = tf.clip_by_value(tf.sqrt(batch_var + eps) * moving_inv,
#                          1 / RMAX,
#                          RMAX)
#     d = tf.clip_by_value((batch_mean - pop_mean) * moving_inv,
#                          -DMAX,
#                          DMAX)
#     scale = tf.stop_gradient(r, name='renorm_r')
#     offset = tf.stop_gradient(d, name='renorm_d')
#     if gamma is not None:
#       scale *= gamma
#       offset *= gamma
#     if beta is not None:
#       offset += beta
#     def batch_statistics():
#       with tf.control_dependencies([train_mean_op, train_var_op]):
#         return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, eps,name=name)
#
#     def population_statistics():
#       return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, gamma, eps,name=name)
#     normed=tf.cond(tf.convert_to_tensor(is_training), batch_statistics, population_statistics)
#     if return_mean:
#       if is_training:
#         return normed, batch_mean
#       else:
#         return normed, pop_mean
#     else:
#       return normed


def Group_batch_norm(x,
               n_out,
               is_training,
                G=2,
               reuse=None,
               gamma=None,
               beta=None,
               axes=[0, 1, 2],
               eps=1e-3,
               scope="bn",
               name="bn_out",
               return_mean=False):
  with tf.variable_scope(scope, reuse=reuse):
    print("use GBN")
    emean = tf.get_variable("ema_mean", [G,n_out], trainable=False)
    evar = tf.get_variable("ema_var", [G,n_out], trainable=False)
    N, C = x.shape
    x = tf.reshape(x, [tf.cast(N//G, tf.int32), tf.cast(G, tf.int32), tf.cast(C, tf.int32)])
    if is_training:
      batch_mean, batch_var = tf.nn.moments(x,[0], name='moments')
      # batch_mean.set_shape([n_out])
      # batch_var.set_shape([n_out])
      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      ema_apply_op_local = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op_local]):
        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
      emean_val = ema.average(batch_mean)
      evar_val = ema.average(batch_var)
      with tf.control_dependencies(
          [tf.assign(emean, emean_val), tf.assign(evar, evar_val)]):
        normed = tf.nn.batch_normalization(
            x, mean, var, beta, gamma, eps, name=name)
        normed= tf.reshape(normed, [tf.cast(N, tf.int32), tf.cast(C, tf.int32)])

    else:
      normed = tf.nn.batch_normalization(
          x, emean, evar, beta, gamma, eps, name=name)
      normed = tf.reshape(normed, [tf.cast(N, tf.int32), tf.cast(C, tf.int32)])
  if return_mean:
    if is_training:
      return normed, mean
    else:
      return normed, emean
  else:
    return normed

def conv2d(input, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', nonlinearity=None, init_scale=1.,
             w_init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    with tf.variable_scope('conv2d'):
      V = tf.get_variable('V', shape=filter_size + [int(input.get_shape()[-1]), num_filters], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
      g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                            initializer=tf.constant_initializer(1.), trainable=True)
      b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.), trainable=True)

      # use weight normalization (Salimans & Kingma, 2016)
      # *表示对应元素相乘，[1, 1, 1, 96]×(3, 3, 3, 96)=(3, 3, 3, 96)每个都乘。
      W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

      print('v_norm', tf.nn.l2_normalize(V, [0, 1, 2]).get_shape())
      print('w', W.get_shape())  # shape=[3, 3, 3, 96]
      x = tf.nn.bias_add(tf.nn.conv2d(input, W, [1] + stride + [1], pad), b)
      print('x', x.get_shape())
      if w_init:  # normalize x
          m_init, v_init = tf.nn.moments(x, [0, 1, 2])
          scale_init = init_scale / tf.sqrt(v_init + 1e-10)
          # `x` will only run after `g, and `b` have executed
          with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
              x = tf.nn.conv2d(input, W, [1]+stride+[1], pad)+ tf.reshape(b, [1, 1, 1, num_filters])

              # x = tf.nn.l2_normalize(x, dim=[0, 1, 2])
              print('x', x.get_shape())

      # apply nonlinearity
      if nonlinearity is not None:
        x = nonlinearity(x)

      return x




def GroupNorm(x,gamma=None,beta=None,G=20,eps=1e-5,reuse=None,scope="GroupNorm",
              name="gn_out",return_mean=False):
  with tf.variable_scope(scope, reuse=reuse):
    N,C=x.shape
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(G,tf.int32),tf.cast(C//G,tf.int32)])
    mean,var=tf.nn.moments(x,[2],keep_dims=True)
    x=(x-mean)/tf.sqrt(var+eps)
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(C,tf.int32)])
    # gamma = tf.Variable(tf.ones(shape=[1,tf.cast(C,tf.int32)]), name="gamma")
    # beta = tf.Variable(tf.zeros(shape=[1,tf.cast(C,tf.int32)]), name="beta")
    if gamma is not None:
      x *= gamma
    if beta is not None:
      x += beta
    normed = tf.identity(x, name=name)
    if return_mean:
        return normed, mean
    else:
        return normed


def batch_norm_mean_only(x,
                         n_out,
                         is_training,
                         reuse=None,
                         gamma=None,
                         beta=None,
                         axes=[0, 1, 2],
                         scope="bnms",
                         name="bnms_out",
                         return_mean=False):
  """Applies mean only batch normalization.
    Collect mean and variances on x except the last dimension. And apply
    normalization as below:
        x_ = gamma * (x - mean) + beta

    Args:
        x: Input tensor, [B, ...].
        n_out: Integer, depth of input variable.
        gamma: Scaling parameter.
        beta: Bias parameter.
        axes: Axes to collect statistics.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Batch-normalized variable.
        mean: Mean used for normalization (optional).
    """
  with tf.variable_scope(scope, reuse=reuse):
    emean = tf.get_variable("ema_mean", [n_out], trainable=False)
    if is_training:
      batch_mean = tf.reduce_mean(x, axes)
      ema = tf.train.ExponentialMovingAverage(decay=0.9)
      ema_apply_op_local = ema.apply([batch_mean])
      with tf.control_dependencies([ema_apply_op_local]):
        mean = tf.identity(batch_mean)
      emean_val = ema.average(batch_mean)
      with tf.control_dependencies([tf.assign(emean, emean_val)]):
        normed = x - batch_mean
      if gamma is not None:
        normed *= gamma
      if beta is not None:
        normed += beta
    else:
      normed = x - emean
      if gamma is not None:
        normed *= gamma
      if beta is not None:
        normed += beta
  if return_mean:
    if is_training:
      return normed, mean
    else:
      return normed, emean
  else:
    return normed
