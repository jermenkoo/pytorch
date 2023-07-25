from typing import List, Any, Dict, Optional

import torch
import torch.fx._pytree as fx_pytree
from torch._export.exported_program import ExportedProgram
from torch._inductor.compile_fx import compile_fx_aot

__all__ = ["aot_compile"]

def aot_compile(
    ep: ExportedProgram,
    example_inputs: List[torch.Tensor],
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Ahead-of-time compile a given FX graph with TorchInductor into a shared library.

    Args:
        ep: The exported graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Path to the generated shared library
    """

    if options is None:
        options = {"aot_from_export": True}
    else:
        options["aot_from_export"] = True

    param_buffer_values = list(ep.state_dict.values())
    flat_example_inputs = fx_pytree.tree_flatten_spec(
        example_inputs, ep.call_spec.in_spec
    )
    all_args = (*param_buffer_values, *flat_example_inputs)

    return compile_fx_aot(
        ep.graph_module,
        all_args,
        config_patches=options,
    )
