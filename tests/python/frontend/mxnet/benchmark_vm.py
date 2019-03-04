"""Benchmarking Relay VM using models from MxNet."""
import numpy as np
import mxnet as mx

import tvm
from tvm.contrib import graph_runtime
from tvm import relay

import model_zoo


def benchmark_execution(mx_symbol,
                        data_shape=(1, 3, 224, 224),
                        out_shape=(1, 1000),
                        dtype='float32'):
    def get_mxnet_output(symbol, x, dtype='float32'):
        from collections import namedtuple
        Batch = namedtuple('Batch', ['data'])
        mod = mx.mod.Module(symbol, label_names=None)
        mod.bind(data_shapes=[('data', x.shape)], for_training=False)
        mod.init_params()
        mod.forward(Batch([mx.nd.array(x.astype(dtype))]))
        out = mod.get_outputs()[0].asnumpy()
        args, auxs = mod.get_params()
        return out, args, auxs

    def get_func_param(symbol, x, args, auxs):
        shape_dict = {"data": x.shape}
        new_sym, params = relay.frontend.from_mxnet(
            symbol, shape_dict, arg_params=args, aux_params=auxs)
        return new_sym, params

    def get_tvm_output(symbol, x, args, auxs, target, ctx, dtype='float32'):
        new_sym, params = get_func_param(symbol, x, args, auxs)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(new_sym, target, params=params)

        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("data", tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    def get_tvm_vm_output(symbol, x, args, auxs, target, ctx, dtype='float32'):
        func, params = get_func_param(symbol, x, args, auxs)
        params = [params[k] for k in params]
        params = [x] + params
        ex = relay.create_executor('vm', mod=relay.Module(), ctx=ctx)
        result = ex.evaluate(func)(*params)
        return result.asnumpy()

    # random input
    x = np.random.uniform(size=data_shape).astype(dtype)
    target = "llvm"
    ctx = tvm.cpu(0)

    _, args, auxs = get_mxnet_output(mx_symbol, x, dtype)
    assert "data" not in args
    tvm_out = get_tvm_output(mx_symbol, x, args, auxs, target, ctx, dtype)
    vm_out = get_tvm_vm_output(mx_symbol, x, args, auxs, target, ctx, dtype)
    tvm.testing.assert_allclose(vm_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_mlp():
    mlp = model_zoo.mx_mlp()
    benchmark_execution(mlp, data_shape=(1, 1, 28, 28), out_shape=(1, 10))


def test_vgg():
    for n in [11, 16]:
        mx_sym = model_zoo.mx_vgg(n)
        benchmark_execution(mx_sym)


def test_resnet():
    for n in [18, 50]:
        mx_sym = model_zoo.mx_resnet(n)
        benchmark_execution(mx_sym)


def test_squeezenet():
    for version in ['1.0', '1.1']:
        mx_sym = model_zoo.mx_squeezenet(version)
        benchmark_execution(mx_sym)


def test_inception_v3():
    shape = {"data": (1, 3, 299, 299)}
    mx_sym = model_zoo.mx_inception_v3()
    benchmark_execution(mx_sym)


def test_dqn():
    shape = {"data": (1, 4, 84, 84)}
    mx_sym = model_zoo.mx_dqn()
    benchmark_execution(mx_sym)


def test_dcgan():
    shape = {"data": (2, 100)}
    mx_sym = model_zoo.mx_dcgan()
    benchmark_execution(mx_sym)


def test_multi_outputs():
    xshape = (10, 27)
    yshape = (10, 9)

    def mx_compose(F, **kwargs):
        x = F.sym.Variable("x")
        y = F.sym.Variable("y")
        z = F.sym.split(x, **kwargs)
        return F.sym.broadcast_sub(F.sym.broadcast_add(z[0], z[2]), y)

    def relay_compose(F, **kwargs):
        x = F.var("x", shape=xshape)
        y = F.var("y", shape=yshape)
        z = F.split(x, **kwargs)
        z = F.subtract(F.add(z[0], z[2]), y)
        return relay.Function(relay.ir_pass.free_vars(z), z)

    mx_sym = mx_compose(mx, num_outputs=3, axis=1)
    benchmark_execution(mx_sym)


if __name__ == '__main__':
    test_mlp()
    # test_resnet()
    # test_vgg()
    # test_multi_outputs()
    # test_dqn()
    # test_dcgan()
    # test_squeezenet()
    # test_inception_v3()
