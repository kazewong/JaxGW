from dataclasses import field

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from jimgw.transforms import (
    BijectiveTransform,
    LogitTransform,
    CartesianToPolarTransform,
    ScaleTransform,
    OffsetTransform,
    ArcSineTransform,
    PowerLawTransform,
    ParetoTransform,
    RayleighTransform,
)


class Prior(Distribution):
    """
    A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    parameter_names: list[str]
    composite: bool = False

    @property
    def n_dim(self) -> int:
        return len(self.parameter_names)

    def __init__(self, parameter_names: list[str]):
        """
        Parameters
        ----------
        parameter_names : list[str]
            A list of names for the parameters of the prior.
        """
        self.parameter_names = parameter_names

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).
        """

        return dict(zip(self.parameter_names, x))

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        raise NotImplementedError

    def log_prob(self, z: dict[str, Array]) -> Float:
        raise NotImplementedError


@jaxtyped(typechecker=typechecker)
class LogisticDistribution(Prior):

    def __repr__(self):
        return f"LogisticDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        self.composite = False
        assert self.n_dim == 1, "LogisticDistribution needs to be 1D distributions"

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a logistic distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        samples = jnp.log(samples / (1 - samples))
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        variable = z[self.parameter_names[0]]
        return -variable - 2 * jnp.log(1 + jnp.exp(-variable))


@jaxtyped(typechecker=typechecker)
class StandardNormalDistribution(Prior):

    def __repr__(self):
        return f"StandardNormalDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs):
        super().__init__(parameter_names)
        self.composite = False
        assert (
            self.n_dim == 1
        ), "StandardNormalDistribution needs to be 1D distributions"

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a standard normal distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.

        """
        samples = jax.random.normal(rng_key, (n_samples,))
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        variable = z[self.parameter_names[0]]
        return -0.5 * variable**2 - 0.5 * jnp.log(2 * jnp.pi)


class SequentialTransformPrior(Prior):
    """
    Transform a prior distribution by applying a sequence of transforms.
    The space before the transform is named as x,
    and the space after the transform is named as z
    """

    base_prior: Prior
    transforms: list[BijectiveTransform]

    def __repr__(self):
        return f"Sequential(priors={self.base_prior}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        base_prior: Prior,
        transforms: list[BijectiveTransform],
    ):

        self.base_prior = base_prior
        self.transforms = transforms
        self.parameter_names = base_prior.parameter_names
        for transform in transforms:
            self.parameter_names = transform.propagate_name(self.parameter_names)
        self.composite = True

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = self.base_prior.sample(rng_key, n_samples)
        return jax.vmap(self.transform)(output)

    def log_prob(self, z: dict[str, Float]) -> Float:
        """
        Evaluating the probability of the transformed variable z.
        This is what flowMC should sample from
        """
        output = 0
        for transform in reversed(self.transforms):
            z, log_jacobian = transform.inverse(z)
            output += log_jacobian
        output += self.base_prior.log_prob(z)
        return output

    def transform(self, x: dict[str, Float]) -> dict[str, Float]:
        for transform in self.transforms:
            x = transform.forward(x)
        return x


class CombinePrior(Prior):
    """
    A prior class constructed by joinning multiple priors together to form a multivariate prior.
    This assumes the priors composing the Combine class are independent.
    """

    base_prior: list[Prior] = field(default_factory=list)

    def __repr__(self):
        return (
            f"Combine(priors={self.base_prior}, parameter_names={self.parameter_names})"
        )

    def __init__(
        self,
        priors: list[Prior],
    ):
        parameter_names = []
        for prior in priors:
            parameter_names += prior.parameter_names
        self.base_prior = priors
        self.parameter_names = parameter_names
        self.composite = True

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        output = {}
        for prior in self.base_prior:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, z: dict[str, Float]) -> Float:
        output = 0.0
        for prior in self.base_prior:
            output += prior.log_prob(z)
        return output


@jaxtyped(typechecker=typechecker)
class UniformPrior(SequentialTransformPrior):
    xmin: float
    xmax: float

    def __repr__(self):
        return f"UniformPrior(xmin={self.xmin}, xmax={self.xmax}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "UniformPrior needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        super().__init__(
            LogisticDistribution([f"{self.parameter_names[0]}_base"]),
            [
                LogitTransform(
                    (
                        [f"{self.parameter_names[0]}_base"],
                        [f"({self.parameter_names[0]}-({xmin}))/{(xmax-xmin)}"],
                    )
                ),
                ScaleTransform(
                    (
                        [f"({self.parameter_names[0]}-({xmin}))/{(xmax-xmin)}"],
                        [f"{self.parameter_names[0]}-({xmin})"],
                    ),
                    xmax - xmin,
                ),
                OffsetTransform(
                    ([f"{self.parameter_names[0]}-({xmin})"], self.parameter_names),
                    xmin,
                ),
            ],
        )


@jaxtyped(typechecker=typechecker)
class SinePrior(SequentialTransformPrior):
    """
    A prior distribution where the pdf is proportional to sin(x) in the range [0, pi].
    """

    def __repr__(self):
        return f"SinePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "SinePrior needs to be 1D distributions"
        super().__init__(
            CosinePrior([f"{self.parameter_names[0]}-pi/2"]),
            [
                OffsetTransform(
                    (
                        (
                            [f"{self.parameter_names[0]}-pi/2"],
                            [f"{self.parameter_names[0]}"],
                        )
                    ),
                    jnp.pi / 2,
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class CosinePrior(SequentialTransformPrior):
    """
    A prior distribution where the pdf is proportional to cos(x) in the range [-pi/2, pi/2].
    """

    def __repr__(self):
        return f"CosinePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str]):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "CosinePrior needs to be 1D distributions"
        super().__init__(
            UniformPrior(-1.0, 1.0, [f"sin({self.parameter_names[0]})"]),
            [
                ArcSineTransform(
                    ([f"sin({self.parameter_names[0]})"], [self.parameter_names[0]])
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class UniformSpherePrior(CombinePrior):

    def __repr__(self):
        return f"UniformSpherePrior(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], max_mag: float = 1.0):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "UniformSpherePrior only takes the name of the vector"
        self.parameter_names = [
            f"{self.parameter_names[0]}_mag",
            f"{self.parameter_names[0]}_theta",
            f"{self.parameter_names[0]}_phi",
        ]
        super().__init__(
            [
                UniformPrior(0.0, max_mag, [self.parameter_names[0]]),
                SinePrior([self.parameter_names[1]]),
                UniformPrior(0.0, 2 * jnp.pi, [self.parameter_names[2]]),
            ]
        )


@jaxtyped(typechecker=typechecker)
class RayleighPrior(SequentialTransformPrior):
    """
    A prior distribution following the Rayleigh distribution with scale parameter sigma.
    """

    sigma: float

    def __repr__(self):
        return f"RayleighPrior(parameter_names={self.parameter_names})"

    def __init__(
        self,
        parameter_names: list[str],
        sigma: float = 1.0,
    ):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "RayleighPrior needs to be 1D distributions"
        self.sigma = sigma
        super().__init__(
            UniformPrior(0.0, 1.0, [f"{self.parameter_names[0]}_base"]),
            [
                RayleighTransform(
                    ([f"{self.parameter_names[0]}_base"], self.parameter_names),
                    sigma=sigma,
                )
            ],
        )


@jaxtyped(typechecker=typechecker)
class PowerLawPrior(SequentialTransformPrior):
    xmin: float
    xmax: float
    alpha: float

    def __repr__(self):
        return f"PowerLawPrior(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        alpha: float,
        parameter_names: list[str],
    ):
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "Power law needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        self.alpha = alpha
        assert self.xmin < self.xmax, "xmin must be less than xmax"
        assert self.xmin > 0.0, "x must be positive"
        if self.alpha == -1.0:
            transform = ParetoTransform(
                ([f"{self.parameter_names[0]}_before_transform"], self.parameter_names),
                xmin,
                xmax,
            )
        else:
            transform = PowerLawTransform(
                ([f"{self.parameter_names[0]}_before_transform"], self.parameter_names),
                xmin,
                xmax,
                alpha,
            )
        super().__init__(
            LogisticDistribution([f"{self.parameter_names[0]}_base"]),
            [
                LogitTransform(
                    (
                        [f"{self.parameter_names[0]}_base"],
                        [f"{self.parameter_names[0]}_before_transform"],
                    )
                ),
                transform,
            ],
        )


def trace_prior_parent(prior: Prior, output: list[Prior] = []) -> list[Prior]:
    if prior.composite:
        if isinstance(prior.base_prior, list):
            for subprior in prior.base_prior:
                output = trace_prior_parent(subprior, output)
        elif isinstance(prior.base_prior, Prior):
            output = trace_prior_parent(prior.base_prior, output)
    else:
        output.append(prior)

    return output


# ====================== Things below may need rework ======================

# @jaxtyped(typechecker=typechecker)
# class Exponential(Prior):
#     """
#     A prior following the power-law with alpha in the range [xmin, xmax).
#     p(x) ~ exp(\alpha x)
#     """

#     xmin: float = 0.0
#     xmax: float = jnp.inf
#     alpha: float = -1.0
#     normalization: float = 1.0

#     def __repr__(self):
#         return f"Exponential(xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha}, naming={self.naming})"

#     def __init__(
#         self,
#         xmin: Float,
#         xmax: Float,
#         alpha: Union[Int, Float],
#         naming: list[str],
#         transforms: dict[str, tuple[str, Callable]] = {},
#         **kwargs,
#     ):
#         super().__init__(naming, transforms)
#         if alpha < 0.0:
#             assert xmin != -jnp.inf, "With negative alpha, xmin must finite"
#         if alpha > 0.0:
#             assert xmax != jnp.inf, "With positive alpha, xmax must finite"
#         assert not jnp.isclose(alpha, 0.0), "alpha=zero is given, use Uniform instead"
#         assert self.n_dim == 1, "Exponential needs to be 1D distributions"

#         self.xmax = xmax
#         self.xmin = xmin
#         self.alpha = alpha

#         self.normalization = self.alpha / (
#             jnp.exp(self.alpha * self.xmax) - jnp.exp(self.alpha * self.xmin)
#         )

#     def sample(
#         self, rng_key: PRNGKeyArray, n_samples: int
#     ) -> dict[str, Float[Array, " n_samples"]]:
#         """
#         Sample from a exponential distribution.

#         Parameters
#         ----------
#         rng_key : PRNGKeyArray
#             A random key to use for sampling.
#         n_samples : int
#             The number of samples to draw.

#         Returns
#         -------
#         samples : dict
#             Samples from the distribution. The keys are the names of the parameters.

#         """
#         q_samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
#         samples = (
#             self.xmin
#             + jnp.log1p(
#                 q_samples * (jnp.exp(self.alpha * (self.xmax - self.xmin)) - 1.0)
#             )
#             / self.alpha
#         )
#         return self.add_name(samples[None])

#     def log_prob(self, x: dict[str, Float]) -> Float:
#         variable = x[self.naming[0]]
#         log_in_range = jnp.where(
#             (variable >= self.xmax) | (variable <= self.xmin),
#             jnp.zeros_like(variable) - jnp.inf,
#             jnp.zeros_like(variable),
#         )
#         log_p = self.alpha * variable + jnp.log(self.normalization)
#         return log_p + log_in_range
