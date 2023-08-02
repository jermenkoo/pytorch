from contextlib import contextmanager
from dataclasses import dataclass

import torch

import torch.utils._pytree as pytree

from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._C._functorch import peek_interpreter_stack, pop_dynamic_layer_stack, push_dynamic_layer_stack
from torch._dynamo.exc import CondOpArgsMismatchError
from torch._functorch.eager_transforms import (
    _unwrap_all_tensors_from_functional,
    _wrap_all_tensors_to_functional,
    functionalize,
)
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)


@contextmanager
def _turn_off_is_fx_tracing():
    _old_is_tracing = torch.fx._symbolic_trace._is_fx_tracing_flag

    try:
        torch.fx._symbolic_trace._is_fx_tracing_flag = False
        yield
    finally:
        torch.fx._symbolic_trace._is_fx_tracing_flag = _old_is_tracing

@contextmanager
def _turn_off_functorch():
    interpreter = peek_interpreter_stack()
    _old_stack = None
    try:
        if interpreter is not None:
            _old_stack = pop_dynamic_layer_stack()
        yield
    finally:
        if _old_stack is not None:
            push_dynamic_layer_stack(_old_stack)


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


def exported_cond(op, *args):
    exclude_keys = (
        DispatchKeySet(DispatchKey.FuncTorchDynamicLayerFrontMode)
        .add(DispatchKey.FuncTorchDynamicLayerBackMode)
        .add(DispatchKey.Functionalize)
    )
    with _turn_off_functorch():
        with _ExcludeDispatchKeyGuard(exclude_keys):
            with torch.utils._python_dispatch._disable_current_modes():
                from torch.fx.experimental.symbolic_shapes import ShapeEnv

                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    shape_env = ShapeEnv()
                    fake_tensor_mode = FakeTensorMode(
                        allow_fallback_kernels=False,
                        allow_non_fake_inputs=True,
                        shape_env=shape_env,
                    )
                else:
                    if fake_tensor_mode.shape_env is None:
                        fake_tensor_mode.shape_env = ShapeEnv()

                def from_fun(t):
                    if isinstance(t, torch.Tensor):
                        from torch._subclasses.fake_tensor import FakeTensor

                        if isinstance(t, FakeTensor):
                            return t
                        return torch.empty_strided(
                            t.size(),
                            t.stride(),
                            device=t.device,
                            dtype=t.dtype,
                            requires_grad=t.requires_grad,
                        )
                    # Need to specialize symbool for now. Won't affect traced graph.
                    elif isinstance(t, torch.SymBool):
                        return t.node._hint
                    elif isinstance(t, torch.fx.proxy.Proxy):
                        raise RuntimeError(
                            f"Unable to symbolically trace HigherOrderOperators {op}"
                        )
                    return t

                args = (args[0], args[1], args[2], tuple(args[3]))
                with fake_tensor_mode:
                    new_args = pytree.tree_map(from_fun, args)

                    # we need to wrap true_fn/false_fn up otherwise the local scope
                    # will contain the original true_fn and false_fn's signature
                    def wrapper(new_args):
                        return cond(*new_args)

                    # need to do it together otherwise will need to merge input ->
                    # duplicated effort with what we have already
                    with _turn_off_is_fx_tracing():
                        expo_result = torch._dynamo.export(
                            wrapper, rewrite_sig=False, fake_mode=fake_tensor_mode
                        )(new_args)
                    gm, guards = expo_result
                    example_inputs = expo_result.example_inputs
                    example_inputs_ids = [id(inp) for inp in example_inputs]
                    id_to_name = {
                        id(guard.obj_weakref()): guard.name
                        for guard in guards
                        if guard.obj_weakref is not None
                        and id(guard.obj_weakref()) in example_inputs_ids
                    }
                    example_names = [id_to_name[id(inp)] for inp in example_inputs]

    # fake the new_args with original args
    local_scope = {"L": {**locals(), "new_args": args}, "G": globals()}
    pos_args = [eval(name, {}, local_scope) for name in example_names]

    # We need to extract true_gm and false_gm from export
    # as export won't add sym bool
    def bind_branch_and_args(gm, pos_args):
        ph2orig = dict(
            zip((ph for ph in gm.graph.nodes if ph.op == "placeholder"), pos_args)
        )
        cond_node = next((n for n in gm.graph.nodes if n.target is cond), None)
        assert cond_node
        true_gm = getattr(gm, cond_node.args[1].name)
        false_gm = getattr(gm, cond_node.args[2].name)
        pos_args = []
        for arg_node in cond_node.args[3]:
            if arg_node.op == "placeholder" and arg_node in ph2orig:
                pos_args.append(ph2orig[arg_node])
            elif arg_node.op == "get_attr":
                pos_args.append(getattr(gm, arg_node.target))
            else:
                raise RuntimeError(f"Cannot bind to original argumentes for {arg_node}")
        return true_gm, false_gm, tuple(pos_args)

    return cond(args[0], *bind_branch_and_args(gm, pos_args))


def cond_compiled(pred, true_fn, false_fn, args):
    if torch._dynamo.is_compiling():
        return cond(pred, true_fn, false_fn, args)
    else:
        return exported_cond(cond, pred, true_fn, false_fn, args)


"""
We're going to define a `cond` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""
cond = HigherOrderOperator("cond")


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    assert isinstance(
        operands, (list, tuple)
    ), "Cond operands must be a list or tuple of tensors"
    assert all(
        isinstance(o, torch.Tensor) for o in operands
    ), "Cond operands must be a list of tensors"

    with disable_proxy_modes_tracing():
        true_graph = make_fx(true_fn)(*operands)
        false_graph = make_fx(false_fn)(*operands)

    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == "output":
            true_outs.extend(node.args)

    for node in false_graph.graph.nodes:
        if node.op == "output":
            false_outs.extend(node.args)

    flat_true_outs, _ = pytree.tree_flatten(true_outs)
    flat_false_outs, _ = pytree.tree_flatten(false_outs)
    if len(flat_true_outs) != len(flat_false_outs):
        raise CondOpArgsMismatchError(
            f"Expected to return same number of outputs but got:"
            f"\n  {true_fn.__name__} returns {len(flat_true_outs)} item(s)"
            f"\n  {false_fn.__name__} returns {len(flat_false_outs)} item(s)"
        )

    for i in range(0, len(flat_true_outs)):
        true_out = flat_true_outs[i]
        false_out = flat_false_outs[i]
        if true_out.meta["tensor_meta"] != false_out.meta["tensor_meta"]:
            raise CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_out.meta['tensor_meta']}"
                f"\n  {false_fn.__name__} returns {false_out.meta['tensor_meta']}"
            )

    # There are probably better ways - I know that create_arg has some self incrementing name
    # magic to it, but since we explicitly have to get the name for register_module,
    # I was not sure how to do that. This kinda simulates it.
    next_name = None
    i = 0
    while not next_name:
        candidate = f"true_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    true_name = next_name
    false_name = f"false_graph_{i}"
    assert not hasattr(proxy_mode.tracer.root, false_name)

    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    args = (pred, true_graph, false_graph, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="conditional"
    )

    # At this point, we're *guaranteed* that whether an output came from the
    # true or false branch is indistinguishable. So, as this is just for tracing
    # purposes, choose the true branch.

    # TODO: Uhh.... it shouldn't matter, but changing this to true_fn results in
    # a FakeTensorMode error :
    # `Current active mode <class 'torch._subclasses.fake_tensor.FakeTensorMode'> not registered`
    out = false_fn(*operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@cond.py_impl(DispatchKey.CompositeExplicitAutograd)
def cond_dense(pred, true_fn, false_fn, operands):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)


cond.py_impl(DispatchKey.Autograd)(autograd_not_implemented(cond, deferred_error=True))


@cond.py_impl(ProxyTorchDispatchMode)
def inner(pred, true_fn, false_fn, operands):
    mode = _get_current_dispatch_mode()
    assert mode is not None, "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            return trace_cond(mode, cond, pred, true_fn, false_fn, operands)
        else:
            return cond(pred, true_fn, false_fn, operands)


@cond.py_impl(FakeTensorMode)
def cond_fake_tensor_mode(pred, true_fn, false_fn, operands):
    true_outs = true_fn(*operands)
    flat_true_outs, _ = pytree.tree_flatten(true_outs)
    flat_false_outs, _ = pytree.tree_flatten(false_fn(*operands))
    if len(flat_true_outs) != len(flat_false_outs):
        raise RuntimeError("Unmatched number of outputs from cond() branches.")

    for true_out, false_out in zip(flat_true_outs, flat_false_outs):
        true_meta = _extract_tensor_metadata(true_out)
        false_meta = _extract_tensor_metadata(false_out)
        if true_meta != false_meta:
            raise CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_meta}"
                f"\n  {false_fn.__name__} returns {false_meta}"
            )
    return true_outs


def _has_potential_branch_input_mutation(branch, inputs):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has mutable op on the input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*inputs)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond is
        # functionalized
        return True
    except Exception as e:
        raise e

    def _detect_input_mutation(gm):
        input_nodes = set()
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                input_nodes.add(node)
            if node.op == "call_function":
                target = node.target
                if (
                    isinstance(target, torch._ops.OpOverload)
                    and target._schema.is_mutable
                ):
                    for arg in node.args:
                        if arg in input_nodes:
                            return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule):
                if _detect_input_mutation(module):
                    return True

        return False

    return _detect_input_mutation(gm)


def _has_potential_branch_input_alias(branch, inputs):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has output aliasing the branch input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*inputs)

    except UnsupportedAliasMutationException:
        # this can happen when nested cond is
        # functionalized
        return True
    except Exception as e:
        raise e

    def _detect_input_alias(gm):
        input_storages = set()
        for node in gm.graph.nodes:
            # We need to check existence of "val" because we reuse the logic here
            # for map operator, where num_mapped_args is a scalar
            # and doesn't have a "val" meta.
            if node.op == "placeholder" and "val" in node.meta:
                input_storages.add(StorageWeakRef(node.meta["val"]._typed_storage()))
            if node.op == "output":

                def check_alias(out):
                    if out is not None and "val" in out.meta:
                        out_storage = StorageWeakRef(out.meta["val"]._typed_storage())
                        return out_storage in input_storages
                    return False

                if any(pytree.tree_flatten(pytree.tree_map(check_alias, node.args))[0]):
                    return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule) and _detect_input_alias(module):
                return True

        return False

    return _detect_input_alias(gm)


@cond.py_impl(DispatchKey.Functionalize)
def cond_func(pred, true_fn, false_fn, inputs):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    unwrapped_inputs = _unwrap_all_tensors_from_functional(
        inputs, reapply_views=reapply_views
    )
    unwrapped_pred = _unwrap_all_tensors_from_functional(
        pred, reapply_views=reapply_views
    )
    mode = "mutations_and_views" if reapply_views else "mutations"
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        functional_true = functionalize(true_fn, remove=mode)
        functional_false = functionalize(false_fn, remove=mode)
        for branch in [true_fn, false_fn]:
            if _has_potential_branch_input_mutation(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch " "might be modifying the input!"
                )

            if _has_potential_branch_input_alias(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch " "might be aliasing the input!"
                )

        cond_return = cond(
            unwrapped_pred, functional_true, functional_false, unwrapped_inputs
        )
        return _wrap_all_tensors_to_functional(cond_return, level=0)


@cond.py_impl(torch._C._functorch.TransformType.Functionalize)
def cond_functionalize(interpreter, pred, true_fn, false_fn, inputs):
    """
    Functionalization implementation for torch.cond. Currently:
      1. We don't allow any input mutation inside the branches
      2. Our check for above condition is not exhaustive
    """
    reapply_views = interpreter.functionalize_add_back_views()
    mode = "mutations_and_views" if reapply_views else "mutations"
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_inputs = _unwrap_all_tensors_from_functional(
        inputs, reapply_views=reapply_views
    )
    unwrapped_pred = _unwrap_all_tensors_from_functional(
        pred, reapply_views=reapply_views
    )

    functional_true_fn = functionalize(true_fn, remove=mode)
    functional_false_fn = functionalize(false_fn, remove=mode)

    with interpreter.lower():
        for branch in [functional_true_fn, functional_false_fn]:
            if _has_potential_branch_input_mutation(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch " "might be modifying the input!"
                )
        for branch in [true_fn, false_fn]:
            if _has_potential_branch_input_alias(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch " "might be aliasing the input!"
                )

        cond_return = cond(
            unwrapped_pred, functional_true_fn, functional_false_fn, unwrapped_inputs
        )
        return _wrap_all_tensors_to_functional(cond_return, level=interpreter.level())


# TODO(voz): Make this automatic for keys, this is very ugly atm
cond.fallthrough(DispatchKey.PythonDispatcher)
cond.fallthrough(DispatchKey.PythonTLSSnapshot)
cond.fallthrough(DispatchKey.ADInplaceOrView)
cond.fallthrough(DispatchKey.BackendSelect)
cond.fallthrough(DispatchKey.AutocastCPU)
