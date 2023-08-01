import contextlib
from typing import Optional

import warnings
import torch
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
    _pop_torch_dispatch_stack, _push_on_torch_dispatch_stack, DispatchKey


# TODO: Limitations and things about enable_torch_dispatch_mode we should fix before exposing it:
# - We need a better user-facing api for _DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)

class TorchDispatchMode:
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """
    def __init__(self, _dispatch_key=None):
        if _dispatch_key is not None:
            assert isinstance(_dispatch_key, torch._C.DispatchKey)
            self.__dict__['_dispatch_key'] = _dispatch_key

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()

    def __enter__(self):
        _push_mode(self, self.__dict__.get("_dispatch_key", None))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode(self.__dict__.get("_dispatch_key", None))

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn("`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`")
        instance = cls(*args, **kwargs)
        return instance

def _get_current_dispatch_mode():
    stack_len = _len_torch_dispatch_stack()
    # Return a user mode on the stack if there are any
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    # Check our proxy mode slot
    mb_proxy = torch._C._get_proxy_tensor_mode()
    if mb_proxy is not None:
        return mb_proxy
    # Check our fake mode slot
    return torch._C._get_fake_tensor_mode()


def _get_current_dispatch_mode_stack():
    stack_len = _len_torch_dispatch_stack()
    user_modes = [_get_dispatch_stack_at(i) for i in range(stack_len)]
    mb_proxy = [] if torch._C._get_proxy_tensor_mode() is None else [torch._C._get_proxy_tensor_mode()]
    mb_fake = [] if torch._C._get_fake_tensor_mode() is None else [torch._C._get_fake_tensor_mode()]
    return user_modes + mb_proxy + mb_fake

def _push_mode(mode, k: Optional[DispatchKey] = None):
    if k is not None:
        from torch._ops import push_mode_for_key, get_cached_ops
        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        # Clear the cache of every op that has been used so far, for this particular key.
        ks = torch._C._functionality_to_backend_keys(k)
        for op in get_cached_ops():
            for key in ks:
                op._uncache_dispatch(key)
        push_mode_for_key(k, mode)
    else:
        _push_on_torch_dispatch_stack(mode)


def _pop_mode(k: Optional[DispatchKey] = None):
    if k is not None:
        from torch._ops import pop_mode_for_key
        return pop_mode_for_key(k)
    else:
        return _pop_torch_dispatch_stack()


@contextlib.contextmanager
def _pop_mode_temporarily(k: Optional[DispatchKey] = None):
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old, k)


@contextlib.contextmanager
def _disable_current_modes():
    mode_len = _len_torch_dispatch_stack()
    old_modes = [_pop_mode() for _ in range(mode_len)]

    # Manually disable proxy and fake modes, if any are active
    mb_proxy = [] if torch._C._get_proxy_tensor_mode() is None else [torch._C._unset_proxy_tensor_mode()]
    mb_fake = [] if torch._C._get_fake_tensor_mode() is None else [torch._C._unset_fake_tensor_mode()]
    try:
        yield old_modes + mb_proxy + mb_fake
    finally:
        for mode in reversed(old_modes):
            _push_mode(mode)
        if mb_proxy:
            torch._C._set_proxy_tensor_mode(mb_proxy[0])
        if mb_fake:
            torch._C._set_fake_tensor_mode(mb_fake[0])


class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

def is_traceable_wrapper_subclass(t):
    # In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,
    # It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.

    is_subclass = isinstance(t, torch.Tensor) and type(t) != torch.Tensor
    return is_subclass and hasattr(t, "__tensor_flatten__") and hasattr(t, "__tensor_unflatten__")

def transform_subclass(t, callback):
    assert is_traceable_wrapper_subclass(t), f"Expects traceable wrapper subclass but got {type(t)}"
    # convert the tensor subclass into its constituent dense tensors,
    # and apply a transformation to each dense tensor.
    flattened_tensors, ctx = type(t).__tensor_flatten__(t)
    transformed_tensors = torch.utils._pytree.tree_map_only(torch.Tensor, callback, flattened_tensors)
    return type(t).__tensor_unflatten__(transformed_tensors, ctx)

def _correct_storage_aliasing(func, args, outs):
    assert isinstance(func, torch._ops.OpOverload)
    assert isinstance(args, (list, tuple))
    assert isinstance(outs, (list, tuple))
    flat_outs, _ = torch.utils._pytree.tree_flatten(outs)
    for x in flat_outs:
        if isinstance(x, torch.Tensor):
            # This is hopefully a reasonable assert:
            # subclasses that rely on this API for output aliasing
            # should always return wrapper tensor subclasses for us to manually alias.
            # in theory if a subclass that needs this API wants to sometimes return
            # plain tensors, we could remove the assert and just not perform the aliasing,
            # but it seems safer to learn more about this case first.
            assert hasattr(x, '__torch_dispatch__')


    def alias_storage(arg, ret):
       # Need to run under no_dispatch, because we explicitly do **not**
       # want our subclass to intercept the set_() call.
       # instead, our subclass should directly have its storage swapped out.
       with torch.utils._mode_utils.no_dispatch():
           if isinstance(ret, torch._subclasses.functional_tensor.FunctionalTensor):
               # directly calling this overload, and passing ret.shape, because we **explicitly**
               # don't want to reset the sizes on ret, if the storage implies a size change.
               # Why?
               # - FunctionalTensor has a CustomSizes, so that it can plumb all calls to metadata to the inner tensor.
               # - TensorImpl.set_sizes_strides() therefore yells if you try to directly change sizes.
               torch.ops.aten.set_.source_Storage_storage_offset(ret, arg.untyped_storage(), 0, ret.shape)
           else:
               ret.set_(arg.untyped_storage())

    def is_match(arg, ret):
        arg_aliases = set() if not arg.alias_info else arg.alias_info.before_set
        out_aliases = set() if not ret.alias_info else ret.alias_info.before_set
        return len(arg_aliases & out_aliases) > 0

    num_args = len(func._schema.arguments)
    num_returns = len(func._schema.returns)
    for arg_idx in range(num_args):
        for return_idx in range(num_returns):
            if is_match(func._schema.arguments[arg_idx], func._schema.returns[return_idx]):
                alias_storage(args[arg_idx], outs[return_idx])

    # Sigh... the torchscript parser has a bug where alias annotations for Tensor[](a) don't show up properly
    # See https://github.com/pytorch/pytorch/issues/106173
    if func.overloadpacket in [
        torch.ops.aten.chunk,
        torch.ops.aten.tensor_split,
        torch.ops.aten.split,
        torch.ops.aten.split_with_sizes,
        torch.ops.aten.hsplit,
        torch.ops.aten.vsplit,
        torch.ops.aten.dsplit,
        torch.ops.aten.unbind,
    ]:
        assert isinstance(outs, list) and all(isinstance(x, torch.Tensor) for x in outs)
        for o in outs:
            # For lists of outputs, need to alias every individual tensor to the input
            alias_storage(args[0], o)

def return_and_correct_aliasing(func, args, out):
    def get_write_alias(x):
        if not x.alias_info or not x.alias_info.before_set:
            return None
        before_set = list(x.alias_info.before_set)
        # torchscript allows for complicated alias sets, but our dispatcher ops only really involve simple aliasing
        assert len(before_set) == 1
        if '!' in list(before_set)[0]:
            return before_set[0]
        return None

    # Fix up the storages of any outs so that they point to the same storage as the input,
    # if func is a view op.
    _correct_storage_aliasing(func, args, [out] if not isinstance(out, (list, tuple)) else out)

    # Next: we need to make sure to return inputs directly, if the output is a mutable alias (e.g. add_()).

    # simple case: none of our outputs have mutable aliases, so we can return the output as-is
    if not any(get_write_alias(r) is not None for r in func._schema.returns):
        return out

    # simplifying assumption: we don't have **any** ops with return types like "-> (Tensor(a!), Tensor)"
    if not all(get_write_alias(r) is not None for r in func._schema.returns):
        raise RuntimeError("Unsupported schema: " + str(func._schema))

    if len(func._schema.returns) == 1:
        arg_idx = get_arg_idx_from_alias(get_write_alias(0))
        return args[arg_idx]

    return [
        args[get_arg_idx_from_alias(get_write_alias(0))] for i in range(len(func._schema.returns))
    ]
