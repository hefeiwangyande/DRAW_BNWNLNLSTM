
import tensorflow as tf
import collections
from tensorflow.python.ops.rnn_cell_impl import  BasicLSTMCell
import nnlib as nn
import math

from layer_norm import layer_norm,layer_norm_mean_only
from batch_norm import batch_norm,batch_renorm,GroupNorm,Group_batch_norm,batch_norm_mean_only


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
Layermean = 0.798
L1_REG_KEY = "L1_REG"
def wn_linear(args, output_size, init_scale=0.1,  wn_init=False,bias_start=0.0,
              ema=None,scope=None):
    total_arg_size = 0
    input_size=0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]
            input_size+=shape[0]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        # if wn_init:
        #     data based initialization of parameters
        #     V = tf.get_variable('V', shape=[total_arg_size, output_size], dtype=dtype,
        #                         initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        #     V_norm = tf.nn.l2_normalize(V.initialized_value(), [1])
        #     x_init = tf.matmul(args[0], V_norm)
        #     m_init, v_init = tf.nn.moments(x_init, [1])
        #     scale_init = init_scale / tf.sqrt(v_init +1e-10)
        #     # g = tf.get_variable('g', dtype=dtype, initializer=scale_init, trainable=True)
        #     # b = tf.get_variable('b', dtype=dtype, initializer=-m_init * scale_init, trainable=True)
        #     x_init = tf.reshape(scale_init, [input_size,1]) * (x_init - tf.reshape(m_init, [input_size,1]))
        #     return x_init
        # else:


            V = tf.get_variable('V',shape=[total_arg_size, output_size], dtype=dtype,
                            initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            if 'j' in scope:
                print("use tanh init")
                g = tf.get_variable('g', shape=[output_size], dtype=tf.float32,
                            initializer=tf.constant_initializer(1.), trainable=True)
            else:
                print("use sigmoid init")
                g = tf.get_variable('g', shape=[output_size], dtype=tf.float32,
                        initializer=tf.constant_initializer(1.), trainable=True)
            # s = tf.get_variable('s', shape=[output_size], dtype=tf.float32,
            #             initializer=tf.constant_initializer(2.178), trainable=True)
            b = tf.get_variable('b', shape=[output_size], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True )
            one_matrix = tf.ones(shape=[total_arg_size, 1],dtype=tf.float32,
                                         )
            # V = V - (1. / total_arg_size) * tf.matmul(one_matrix, tf.matmul(tf.transpose(one_matrix), V))
            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(args[0], V)
            scaler = g/tf.sqrt(tf.reduce_sum(tf.square(V),[0]))
            x = tf.reshape(scaler,[1,output_size])*x + tf.reshape(b,[1,output_size])
            return x


def _linear(args, output_size, bias, init_scale=0.1, bias_start=0.0,
            scope=None):
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "Matrix", [total_arg_size, output_size],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(
            -init_scale, init_scale, dtype=dtype))
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(nn.concat(args, 1), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term
class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype
'''layer normalization'''
def ln(tensor, scope=None, epsilon=1e-5):
  assert(len(tensor.get_shape()) == 2)
  m, v = tf.nn.moments(tensor, [1], keep_dims=True)
  if not isinstance(scope, str):
    scope = ''
  with tf.variable_scope(scope+'layer_norm'):
    scale = tf.get_variable(name='scale',
                            shape=[tensor.get_shape()[1]],
                            initializer=tf.constant_initializer(1))
    shift = tf.get_variable('shift',
                            [tensor.get_shape()[1]],
                            initializer=tf.constant_initializer(0))
  LN_initial = (tensor - m) / tf.sqrt(v + epsilon)
  return LN_initial*scale + shift
class BasicLSTMCell(BasicLSTMCell):
  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               ln=False,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    super(BasicLSTMCell, self).__init__(num_units=num_units)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._ln=ln
    if activation:
      self._activation = activation
    else:
      self._activation = tf.tanh

  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)
    input_depth = inputs_shape[-1]
    h_depth = self._num_units
    print("start constract weight")
    self._kernel = tf.get_variable(
        "weight",
        shape=[input_depth+h_depth, 4 * self._num_units])
    if self._ln:
        print("use LN")
        self._kernel = tf.get_variable(
            "weight_x",
            shape=[input_depth , 4 * self._num_units])
    self.h_kernel = tf.get_variable(
        "weight_h",
        shape=[ h_depth, 4 * self._num_units])
    self._bias = tf.get_variable(
        "bais",
        shape=[4 * self._num_units],
        initializer=tf.zeros_initializer())
    self.built = True


  def call(self, inputs, state):
    """Long short-term memory cell (LSTM).
    Args:
      inputs: `2-D` tensor with shape `[batch_size, input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size, num_units]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size, 2 * num_units]`.
    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = tf.sigmoid
    one = tf.constant(1, dtype=tf.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = tf.split(value=state, num_or_size_splits=2, axis=one)

    if self._ln:
        gate_x_inputs = tf.matmul(inputs, self._kernel)
        gate_x_inputs =ln(gate_x_inputs, scope='x_input')
        gate_h_inputs = tf.matmul(h, self.h_kernel)
        gate_h_inputs = ln(gate_h_inputs, scope='h_input')
        gate_inputs=tf.add(gate_x_inputs,gate_h_inputs)
    else:
        gate_inputs=tf.matmul(tf.concat([inputs,h],1), self._kernel)
    gate_inputs = tf.add(gate_inputs, self._bias)
    # [N,self._num_units*4]
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor =tf.constant(self._forget_bias, dtype=f.dtype)
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = tf.add
    multiply = tf.multiply
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
    if self._ln:
        new_c=ln(new_c, scope='new_c/')
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = tf.concat([new_c, new_h], 1)
    return new_h, new_state



class LSTMCell(BasicLSTMCell):

  def __init__(self,
               num_units,
               forget_bias=0.0,
               input_size=None,
               init_scale=0.1,
               state_is_tuple=True,
               eps=1e-3,
               affine=True,
               keep_prob=1.0,
               gate_activation=tf.sigmoid,
               state_activation=tf.tanh,
               l1_reg=0.0):

    super(LSTMCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        state_is_tuple=state_is_tuple)
    self.eps = eps
    self.affine = affine
    self.init_scale = init_scale
    self.unroll_count = -1
    self.keep_prob = keep_prob
    self.gate_activation = gate_activation
    self.state_activation = state_activation
    self.l1_reg = l1_reg

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    self.unroll_count += 1
    with tf.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = nn.split(state, 2, 1)
      with tf.variable_scope("LSTM_weights", reuse=reuse):
        print("resue is " ,reuse)
        i2h = _linear(
            [inputs],
            4 * self._num_units,
            True,
            scope="LinearI",
            init_scale=self.init_scale)
        h2h = _linear(
            [h],
            4 * self._num_units,
            True,
            scope="LinearH",
            init_scale=self.init_scale)
      i, j, f, o = nn.split(i2h + h2h, 4, 1)
      new_c = (c * self.gate_activation(f + self._forget_bias) +
               self.gate_activation(i) * self.state_activation(j))

      new_h = self.state_activation(new_c) * self.gate_activation(o)
      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = nn.concat([new_c, new_h], 1)
    return new_h, new_state









class LSTMBNCell(BasicLSTMCell):

  def __init__(self,
               num_units,
               forget_bias=0.0,
               input_size=None,
               init_scale=0.1,
               state_is_tuple=True,
               eps=1e-3,
               affine=True,
               keep_prob=1.0,
               gate_activation=tf.sigmoid,
               state_activation=tf.tanh,
               l1_reg=0.0):

    super(LSTMBNCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        state_is_tuple=state_is_tuple)
    self.eps = eps
    self.affine = affine
    self.init_scale = init_scale
    self.unroll_count = -1
    self.keep_prob = keep_prob
    self.gate_activation = gate_activation
    self.state_activation = state_activation
    self.l1_reg = l1_reg

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    self.unroll_count += 1
    with tf.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = nn.split(state, 2, 1)
      with tf.variable_scope("LSTM_weights", reuse=reuse):
        print("resue is " ,reuse)
        i2h = _linear(
            [inputs],
            4 * self._num_units,
            True,
            scope="LinearI",
            init_scale=self.init_scale)
        h2h = _linear(
            [h],
            4 * self._num_units,
            True,
            scope="LinearH",
            init_scale=self.init_scale)
        beta_i = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_i")
        gamma_i = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.1},
            name="gamma_i")
        beta_h = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_h")
        gamma_h = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.1},
            name="gamma_h")
        beta_c = nn.weight_variable(
            [self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_c")
        gamma_c = nn.weight_variable(
            [self._num_units],
            init_method="constant",
            init_param={"val": 0.1},
            name="gamma_c")
      i2h_norm, mean_i =batch_norm(
          i2h,
          self._num_units * 4,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_i,
          beta=beta_i,
          axes=[0],
          eps=self.eps,
          scope="bn_i_{}".format(self.unroll_count),
          return_mean=True)
      # if self.l1_reg > 0.0:
        # tf.add_to_collection(L1_REG_KEY,
                             # self.l1_reg * tf.reduce_mean(tf.abs(i2h - mean_i)))
      h2h_norm, mean_h = batch_norm(
          h2h,
          self._num_units * 4,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_h,
          beta=beta_h,
          axes=[0],
          eps=self.eps,
          scope="bn_h_{}".format(self.unroll_count),
          return_mean=True)
      # if self.l1_reg > 0.0:
        # tf.add_to_collection(L1_REG_KEY,
        #                      self.l1_reg * tf.reduce_mean(tf.abs(h2h - mean_h)))
      i, j, f, o = nn.split(i2h_norm + h2h_norm, 4, 1)
      new_c = (c * self.gate_activation(f + self._forget_bias) +
               self.gate_activation(i) * self.state_activation(j))
      new_c_norm, mean_c = batch_norm(
          new_c,
          self._num_units,
          is_training,
          reuse=reuse_bn,
          gamma=gamma_c,
          beta=beta_c,
          axes=[0],
          eps=self.eps,
          scope="bn_c_{}".format(self.unroll_count),
          return_mean=True)
      # if self.l1_reg > 0.0:
        # tf.add_to_collection(L1_REG_KEY, self.l1_reg *
        #                      tf.reduce_mean(tf.abs(new_c - mean_c)))
      new_h = self.state_activation(new_c_norm) * self.gate_activation(o)
      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c_norm, new_h)
      else:
        new_state = nn.concat([new_c_norm, new_h], 1)
    return new_h, new_state

class LSTMLNCell(BasicLSTMCell):

  def __init__(self,
               num_units,
               forget_bias=0.0,
               input_size=None,
               init_scale=0.1,
               state_is_tuple=True,
               eps=1e-3,
               affine=True,
               keep_prob=1.0,
               gate_activation=tf.sigmoid,
               state_activation=tf.tanh,
               l1_reg=0.0):

    super(LSTMLNCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        state_is_tuple=state_is_tuple)
    self.eps = eps
    self.affine = affine
    self.init_scale = init_scale
    self.unroll_count = -1
    self.keep_prob = keep_prob
    self.gate_activation = gate_activation
    self.state_activation = state_activation
    self.l1_reg = l1_reg

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    self.unroll_count += 1
    with tf.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = nn.split(state, 2, 1)
      with tf.variable_scope("LSTM_weights", reuse=reuse):
        print("resue is " ,reuse)
        i2h = _linear(
            [inputs],
            4 * self._num_units,
            True,
            scope="LinearI",
            init_scale=self.init_scale)
        h2h = _linear(
            [h],
            4 * self._num_units,
            True,
            scope="LinearH",
            init_scale=self.init_scale)
        beta_i = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_i")
        gamma_i = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.1},
            name="gamma_i")
        beta_h = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_h")
        gamma_h = nn.weight_variable(
            [4 * self._num_units],
            init_method="constant",
            init_param={"val": 0.1},
            name="gamma_h")
        beta_c = nn.weight_variable(
            [self._num_units],
            init_method="constant",
            init_param={"val": 0.0},
            name="beta_c")
        gamma_c = nn.weight_variable(
            [self._num_units],
            init_method="constant",
            init_param={"val": 0.1},
            name="gamma_c")
      i2h_norm, mean_i =layer_norm(
          i2h,
          gamma=gamma_i,
          beta=beta_i,
          axes=[1],
          eps=self.eps,
          scope="ln_i_{}".format(self.unroll_count),
          return_mean=True)
      # if self.l1_reg > 0.0:
        # tf.add_to_collection(L1_REG_KEY,
                             # self.l1_reg * tf.reduce_mean(tf.abs(i2h - mean_i)))
      h2h_norm, mean_h = layer_norm(
          h2h,
          gamma=gamma_h,
          beta=beta_h,
          axes=[1],
          eps=self.eps,
          scope="ln_h_{}".format(self.unroll_count),
          return_mean=True)
      # if self.l1_reg > 0.0:
        # tf.add_to_collection(L1_REG_KEY,
        #                      self.l1_reg * tf.reduce_mean(tf.abs(h2h - mean_h)))
      i, j, f, o = nn.split(i2h_norm + h2h_norm, 4, 1)
      new_c = (c * self.gate_activation(f + self._forget_bias) +
               self.gate_activation(i) * self.state_activation(j))
      new_c_norm, mean_c = layer_norm(
          new_c,
          gamma=gamma_c,
          beta=beta_c,
          axes=[1],
          eps=self.eps,
          scope="ln_c_{}".format(self.unroll_count),
          return_mean=True)
      # if self.l1_reg > 0.0:
        # tf.add_to_collection(L1_REG_KEY, self.l1_reg *
        #                      tf.reduce_mean(tf.abs(new_c - mean_c)))
      new_h = self.state_activation(new_c_norm) * self.gate_activation(o)
      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c_norm, new_h)
      else:
        new_state = nn.concat([new_c_norm, new_h], 1)
    return new_h, new_state

class LSTMWNCell(BasicLSTMCell):

  def __init__(self,
               num_units,
               forget_bias=0.0,
               input_size=None,
               init_scale=0.1,
               state_is_tuple=True,
               activation=tf.tanh,
               dropout_after=False,
               eps=1e-3,
               affine=True,
               gate_activation=tf.sigmoid,
               state_activation=tf.tanh,
               l1_reg=0.0):
    super(LSTMWNCell, self).__init__(
        num_units,
        forget_bias=forget_bias,
        state_is_tuple=state_is_tuple)
    self.eps = eps
    self.affine = affine
    self.init_scale = init_scale
    self.gate_activation = gate_activation
    self.state_activation = state_activation
    self.l1_reg = l1_reg
    self.unroll_count = -1

  def __call__(self,
               inputs,
               state,
               scope=None,
               is_training=True,
               reuse=None,
               reuse_bn=None):
    self.unroll_count += 1
    with tf.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = tf.split(state, 2, 1)
      with tf.variable_scope("LSTM_weights", reuse=reuse):
        i2h = wn_linear(
            [inputs],
            4* self._num_units,
            scope="LinearI",
            init_scale=self.init_scale,
            )
        h2h = wn_linear(
            [h],
            4 * self._num_units,
            scope="LinearH",
            init_scale=self.init_scale,
        )

      # if self.l1_reg > 0.0:
          # tf.add_to_collection(L1_REG_KEY,
          #                      self.l1_reg * tf.reduce_mean(tf.abs(i2h) - Layermean))
      # if self.l1_reg > 0.0:
      #     tf.add_to_collection(L1_REG_KEY,self.l1_reg*tf.reduce_mean(tf.abs(h2h) - Layermean))


      i, j, f, o = tf.split(i2h+h2h, 4, 1)
      new_c = (c * self.gate_activation(f + self._forget_bias) +
               self.gate_activation(i) * self.state_activation(j))
      new_h = self.state_activation(new_c) * self.gate_activation(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat([new_c, new_h], 1)
    return new_h, new_state