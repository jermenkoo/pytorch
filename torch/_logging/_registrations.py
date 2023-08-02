from ._internal import register_artifact, register_log

register_log("dynamo", "torch._dynamo")
register_log("aot", "torch._functorch.aot_autograd")
register_log("inductor", "torch._inductor")
register_log("dynamic", "torch.fx.experimental.symbolic_shapes")
register_log("torch", "torch")
register_log("distributed", "torch.distributed")

register_artifact("guards")
register_artifact("bytecode", off_by_default=True)
register_artifact("graph")
register_artifact("graph_code")
register_artifact("graph_sizes")
register_artifact("trace_source")
register_artifact("trace_call")
register_artifact("aot_graphs")
register_artifact("aot_joint_graph")
register_artifact("ddp_graphs")
register_artifact("recompiles")
register_artifact("graph_breaks")
register_artifact("not_implemented")
register_artifact("output_code", off_by_default=True)
register_artifact("schedule", off_by_default=True)
register_artifact("perf_hints", off_by_default=True)

register_artifact("custom_format_test_artifact", log_format="")
