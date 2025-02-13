from jimgw.prior import *
from jimgw.utils import trace_prior_parent
import scipy.stats as stats


class TestUnivariatePrior:
    def test_logistic(self):
        p = LogisticDistribution(["x"])
        # Check that the log_prob is finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Cross-check log_prob with scipy.stats.logistic
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(jax.vmap(p.log_prob)(p.add_name(x[None])), stats.logistic.logpdf(x))

    def test_standard_normal(self):
        p = StandardNormalDistribution(["x"])
        # Check that the log_prob is finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Cross-check log_prob with scipy.stats.norm
        x = jnp.linspace(-10.0, 10.0, 1000)
        assert jnp.allclose(jax.vmap(p.log_prob)(p.add_name(x[None])), stats.norm.logpdf(x))

    def test_uniform(self):
        xmin, xmax = -10.0, 10.0
        p = UniformPrior(xmin, xmax, ["x"])
        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x']))
        # Check that the log_prob is correct in the support
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.allclose(log_prob, jnp.log(1.0 / (xmax - xmin)))

    def test_sine(self):
        p = SinePrior(["x"])
        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x']))
        # Check that the log_prob is finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Check that the log_prob is correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior.base_prior.transform)(x)
        y = jax.vmap(p.base_prior.transform)(y)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.sin(y['x'])/2.0))
        
    def test_cosine(self):
        p = CosinePrior(["x"])
        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x']))
        # Check that the log_prob is finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Check that the log_prob is correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(-10.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior.transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), jnp.log(jnp.cos(y['x'])/2.0))

    def test_uniform_sphere(self):
        p = UniformSpherePrior(["x"])
        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x_mag']))
        assert jnp.all(jnp.isfinite(samples['x_theta']))
        assert jnp.all(jnp.isfinite(samples['x_phi']))
        # Check that the log_prob is finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))

    def test_power_law(self):
        def powerlaw_log_pdf(x, alpha, xmin, xmax):
            if alpha == -1.0:
                normalization = 1./(jnp.log(xmax) - jnp.log(xmin))
            else:
                normalization = (1.0 + alpha) / (xmax**(1.0 + alpha) - xmin**(1.0 + alpha))
            return jnp.log(normalization) + alpha * jnp.log(x)
        
        def func(alpha):
            xmin = 0.05
            xmax = 10.0
            p = PowerLawPrior(xmin, xmax, alpha, ["x"])
            # Check that all the samples are finite
            powerlaw_samples = p.sample(jax.random.PRNGKey(0), 10000)
            assert jnp.all(jnp.isfinite(powerlaw_samples['x']))
            
            # Check that all the log_probs are finite
            log_p = jax.vmap(p.log_prob, [0])(powerlaw_samples)
            assert jnp.all(jnp.isfinite(log_p))
            
            # Check that the log_prob is correct in the support
            log_prob = jax.vmap(p.log_prob)(powerlaw_samples)
            standard_log_prob = powerlaw_log_pdf(powerlaw_samples['x'], alpha, xmin, xmax)
            # log pdf of powerlaw
            assert jnp.allclose(log_prob, standard_log_prob, atol=1e-4)

        # Test Pareto Transform
        func(-1.0)
        # Test other values of alpha
        print("Testing PowerLawPrior")
        positive_alpha = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for alpha_val in positive_alpha:
            func(alpha_val)
        negative_alpha = [-0.5, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0]
        for alpha_val in negative_alpha:
            func(alpha_val)

    def test_Rayleigh(self):
        p = RayleighPrior(["x"])
        # Check that all the samples are finite
        samples = p.sample(jax.random.PRNGKey(0), 10000)
        assert jnp.all(jnp.isfinite(samples['x']))
        # Check that the log_prob is finite
        log_prob = jax.vmap(p.log_prob)(samples)
        assert jnp.all(jnp.isfinite(log_prob))
        # Check that the log_prob is correct in the support
        x = trace_prior_parent(p, [])[0].add_name(jnp.linspace(0.0, 10.0, 1000)[None])
        y = jax.vmap(p.base_prior.transform)(x)
        y = jax.vmap(p.transform)(y)
        assert jnp.allclose(jax.vmap(p.log_prob)(y), stats.rayleigh.logpdf(y['x']))
