import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import UniformPrior, RayleighPrior, CombinePrior
from jimgw.transforms import PeriodicTransform, BoundToUnbound
from jimgw.base import LikelihoodBase
from jaxtyping import Float

jax.config.update("jax_enable_x64", True)

prior = CombinePrior(
    [
        RayleighPrior(["test_r"]),
        UniformPrior(0.0, 2.0 * jnp.pi, ["test"]),
    ]
)
sample_transforms = [
    PeriodicTransform(
        name_mapping=[["test_r", "test"], ["test_x", "test_y"]],
        xmin=0.0,
        xmax=2.0 * jnp.pi,
    )
]

# Uncomment the following code to use a uniform prior without periodicity
# prior = UniformPrior(0.0, 2.0 * jnp.pi, ["test"])
# sample_transforms = [
#     BoundToUnbound(name_mapping = [["test"], ["test_unbound"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi)
# ]

likelihood_transforms = []


class CosLikelihood(LikelihoodBase):

    def __init__(self):
        pass

    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        return jnp.log((jnp.cos(params["test"]) + 1.0) / 2.0 / jnp.pi)


likelihood = CosLikelihood()

mass_matrix = jnp.eye(prior.n_dim)

n_local_steps = 1_000

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=1,
    n_loop_production=1,
    n_local_steps=n_local_steps,
    n_global_steps=0,
    n_chains=1,
    n_epochs=1,
    learning_rate=1e-4,
    n_max_examples=10000,
    n_flow_sample=1,
    momentum=0.9,
    batch_size=100,
    use_global=False,
    train_thinning=1,
    output_thinning=1,
    local_sampler_arg={"step_size": mass_matrix * 2e-1},
)

jim.sample(jax.random.PRNGKey(12345))
jim.print_summary()
samples = jim.get_samples()

# Uncomment the following code to plot the walker history and histogram
# import matplotlib.pyplot as plt

# plt.plot(samples["test"])
# plt.ylim(0.0, 2.0 * jnp.pi)
# plt.xlim(0, 300)
# if prior.n_dim == 1:
#     plt.savefig("walker_history_uniform.jpg")
# else:
#     plt.savefig("walker_history_periodic.jpg")
# plt.close()

# plt.hist(samples["test"], label="Samples", density=True, bins=50)
# x = jnp.linspace(0.0, 2.0 * jnp.pi, 1000)
# y = (jnp.cos(x) + 1.0) / 2.0 / jnp.pi
# plt.plot(x, y, label="Likelihood")
# plt.ylim(0.0)
# plt.xlim(0.0, 2.0 * jnp.pi)
# plt.legend()
# if prior.n_dim == 1:
#     plt.savefig("histogram_uniform.jpg")
# else:
#     plt.savefig("histogram_periodic.jpg")
# plt.close()
