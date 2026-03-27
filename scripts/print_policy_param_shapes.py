import jax
import jax.numpy as jnp

from JaxCQL.jax_utils import init_rng, next_rng
from JaxCQL.model import TanhGaussianPolicy


init_rng(0)
policy = TanhGaussianPolicy(7, 7, "256-256", True, 1.0, -1.0)
params = policy.init(next_rng(policy.rng_keys()), jnp.zeros((1, 7)))
print(jax.tree_util.tree_map(lambda x: getattr(x, "shape", None), params))
