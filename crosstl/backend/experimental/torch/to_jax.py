import jax
import jax.numpy as jnp
from typing import Any, Dict


FX_TO_JAX_OPS = {
    "add": "jnp.add",
    "mul": "jnp.multiply",
    "div": "jnp.divide",
    "sub": "jnp.subtract",
    "tensor": "jnp.array",
    "cat": "jnp.concatenate",
    "reshape": "jnp.reshape",
}


def fx_to_jax(graph_module: fx.GraphModule) -> str:
    """Convert a torch.fx GraphModule to JAX code, avoiding constant folding properly."""
    jax_code = []
    jax_code.append("import jax")
    jax_code.append("import jax.numpy as jnp")
    jax_code.append("\n\ndef jax_function(pos):")

    # Track input variables and intermediate results
    var_map: Dict[str, str] = {}
    dynamic_input_index = 0  # Track dynamic inputs for pos

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            # Input tensor for JAX
            jax_code.append(f"    {node.name} = pos  # Input placeholder")
            var_map[node.name] = node.name

        elif node.op == "get_attr":
            # Handle constant tensors correctly
            tensor_value = getattr(graph_module, node.target)

            if isinstance(tensor_value, torch.Tensor):
                # Treat as a constant tensor if small, otherwise inject dynamically
                if tensor_value.numel() < 10:  # Treat small tensors as constants
                    jax_code.append(
                        f"    {node.name} = jnp.array({tensor_value.tolist()}, dtype=jnp.int32)  # Constant tensor"
                    )
                    var_map[node.name] = node.name
                else:
                    # Inject large tensors dynamically
                    jax_code.append(
                        f"    {node.name} = pos[{dynamic_input_index + 1}]  # Dynamic tensor as input"
                    )
                    var_map[node.name] = node.name
                    dynamic_input_index += 1

        elif node.op == "call_function":
            # Convert Torch function to JAX equivalent
            jax_func = FX_TO_JAX_OPS.get(node.target.__name__, None)

            if jax_func is None:
                raise ValueError(f"Unsupported operation: {node.target.__name__}")

            args = ", ".join(var_map.get(str(arg), str(arg)) for arg in node.args)
            jax_code.append(f"    {node.name} = {jax_func}({args})")
            var_map[node.name] = node.name

        elif node.op == "call_method":
            method_name = node.target
            base_var = var_map.get(str(node.args[0]), str(node.args[0]))

            if method_name == "reshape":
                shape = node.args[1]
                jax_code.append(f"    {node.name} = {base_var}.reshape({shape})")
                var_map[node.name] = node.name

        elif node.op == "output":
            # Correctly return the output without indexing
            output_name = var_map.get(str(node.args[0]), str(node.args[0]))
            jax_code.append(f"    return {output_name}")

    return "\n".join(jax_code)
