import tensorflow as tf
import numpy as np
from tvm import relay
from tvm.relay.frontend.tensorflow import from_tensorflow

def ex1():
    tf.reset_default_graph()
    i = tf.constant(0)
    c = lambda i: tf.less(i,10)
    b = lambda i: tf.add(i,1)
    r = tf.while_loop(c, b, [i])

    graph = tf.get_default_graph()

    with tf.Session() as sess:
        tf_out = sess.run(r)

    expr, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    ex = relay.create_executor('debug')
    relay_out = ex.evaluate(expr)(**params)
    np.testing.assert_allclose(tf_out, relay_out.asnumpy())


def ex2():
    tf.reset_default_graph()
    i0 = tf.constant(0)
    j0 = tf.ones([2,2])
    c = lambda i, j: i < 10
    b = lambda i, j: [tf.add(i,1), j]
    r = tf.while_loop(c, b, loop_vars=[i0, j0])

    graph = tf.get_default_graph()

    with tf.Session() as sess:
        tf_out = sess.run(r)

    expr, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    ex = relay.create_executor('debug')
    relay_out = ex.evaluate(expr)(**params)
    np.testing.assert_allclose(tf_out, relay_out.asnumpy())

if __name__ == "__main__":
    ex1()
    ex2()
