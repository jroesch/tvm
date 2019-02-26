import tensorflow as tf
import numpy as np
from tvm import relay
from tvm.relay.frontend.tensorflow import from_tensorflow

def check_equal(graph, tf_out):
    expr, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    ex = relay.create_executor('debug')
    relay_out = ex.evaluate(expr)(**params)
    if isinstance(relay_out, relay.backend.interpreter.TensorValue):
        np.testing.assert_allclose(tf_out, relay_out.asnumpy())
    else:
        if not isinstance(tf_out, list):
            tf_out = [tf_out]
        for x, y in zip(tf_out, [r.asnumpy() for r in relay_out]):
            np.testing.assert_allclose(x, y)

def ex1():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.constant(0)
        c = lambda i: tf.less(i,10)
        b = lambda i: tf.add(i,1)
        r = tf.while_loop(c, b, [i])

        with tf.Session() as sess:
            tf_out = sess.run(r)

        check_equal(graph, tf_out)


def ex2():
    graph = tf.Graph()
    with graph.as_default():
        i0 = tf.constant(0)
        j0 = tf.ones([2,2])
        c = lambda i, j: i < 10
        b = lambda i, j: [tf.add(i,1), j]
        i1, i2 = tf.while_loop(c, b, loop_vars=[i0, j0])
        i1 += tf.constant(1337)

        with tf.Session() as sess:
            tf_out = sess.run(i1)

    check_equal(graph, tf_out)

def ex3():
    graph = tf.Graph()
    with graph.as_default():
        i0 = tf.constant(0)
        j0 = tf.ones([2,2])
        k0 = tf.constant(4)
        c = lambda i, j, k: i < 10
        b = lambda i, j, k: [tf.add(i,1), j, k + i]
        r = tf.while_loop(c, b, loop_vars=[i0, j0, k0])

        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)

if __name__ == "__main__":
    ex1()
    ex2()
    ex3()
