# Import GW packages
import numpy as np
from lal import GreenwichMeanSiderealTime
from ripple import ms_to_Mc_eta
from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jaxgw.PE.detector_preset import *
from jaxgw.PE.single_event_likelihood import single_detector_likelihood
from jaxgw.PE.detector_projection import make_detector_response, get_detector_response
from jaxgw.PE.generate_noise import generate_noise
import matplotlib.pyplot as plt
import corner
from multiprocessing import Pool

# from jaxgw.PE.utils import inner_product
# from jax import grad, vmap
# from functools import partial

# Import FlowMC stuff
# from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
# from flowMC.nfmodel.rqSpline import RQSpline
# from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys

import emcee
import jax
import jax.numpy as jnp


from multiprocessing import cpu_count

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))



#########################################
# Setting up the GW likleilhood
#########################################

# Detector Setup

f_sampling = 2048
duration = 4
fmin = 20
ifos = ["H1", "L1", "V1"]

freqs, psd_dict, noise_dict = generate_noise(1234, f_sampling, duration, fmin, ifos)


H1 = get_H1()
L1 = get_L1()
V1 = get_V1()

f_ref = fmin
trigger_time = 1126259462.4
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)

# True parameters for injection

# Mc, eta, chi1, chi2, dist, tc, phic, inclination, polarization_angle, ra, dec,
m1 = 30
m2 = 25
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
distance = 1600
true_params = jnp.array(
    [Mc, eta, 0.3, -0.4, distance, 0.0, 0.0, np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3]
)

# We now can generate data

f_list = freqs[freqs > fmin]
psd_list = [psd_dict["H1"][freqs > fmin], psd_dict["L1"][freqs > fmin]]
waveform_generator = lambda f_, theta_: gen_IMRPhenomD_polar(f_, theta_, f_ref)

H1_signal = get_detector_response(
    waveform_generator, true_params, f_list, H1, gmst, epoch
)
H1_noise_psd = noise_dict["H1"][freqs > fmin]
H1_data = H1_noise_psd + H1_signal

L1_signal = get_detector_response(
    waveform_generator, true_params, f_list, L1, gmst, epoch
)
L1_noise_psd = noise_dict["L1"][freqs > fmin]
L1_data = L1_noise_psd + L1_signal


def logL(p):
    # Adding on the true ones
    extrinsic_variables = jnp.array(
        [np.pi / 3, np.pi / 3, np.pi / 3, np.pi / 3]
    )
    params = jnp.concatenate((p, extrinsic_variables))
    logL_H1 = single_detector_likelihood(
        waveform_generator, params, H1_data, f_list, psd_list[0], H1, gmst, epoch
    )
    logL_L1 = single_detector_likelihood(
        waveform_generator, params, L1_data, f_list, psd_list[1], L1, gmst, epoch
    )

    return logL_H1 + logL_L1

prior_range = jnp.array(
    [
        [10, 80],
        [0.09, 0.25],
        [-1, 1],
        [-1, 1],
        [400,2000],
        [-0.1, 0.1],
        [0, 2*np.pi],
    ]
)

def top_hat(x):
    output = 0.0
    for i in range(n_dim):
        output = jax.lax.cond(
            x[i] >= prior_range[i, 0], lambda: output, lambda: -jnp.inf
        )
        output = jax.lax.cond(
            x[i] <= prior_range[i, 1], lambda: output, lambda: -jnp.inf
        )
    return output

def posterior(theta):
    # print(theta)
    prior = top_hat(theta)
    if theta[1] > 0.25 or theta[0] < 0.0:
        L =  -jnp.inf
    else:
        L = logL(theta)
    # print(L, prior)
    return L + prior


n_dim, n_chains = 7, 30
nsteps = int(1e3)
ivar = 1. / np.random.rand(n_dim)

true_params = jnp.array([Mc, eta, 0.3, -0.4])
rng_key_set = initialize_rng_keys(n_chains, seed=41)

initial_noise = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
initial_mean = jnp.array([Mc, eta, 0.3, -0.4, distance, 0, 0. ]).reshape(1,7)

sigma = jnp.array([1, 0.001, 0.1, 0.1, 10, 0.001, 0.01]).reshape(1,7)

initial_position = np.array(initial_mean + sigma * initial_noise)
initial_position[initial_position[:,1]>0.25,1] = 0.25
initial_position[initial_position[:,6]<0.,6] = 0.0
initial_position = jnp.array(initial_position)

# sampler = emcee.EnsembleSampler(n_chains, n_dim, posterior)#, args=[ivar])
# sampler.run_mcmc(initial_position, nsteps,  skip_initial_state_check=True, progress=True)

from multiprocessing import get_context

with get_context("fork").Pool() as pool:
    sampler = emcee.EnsembleSampler(n_chains, n_dim, posterior, pool=pool)
    sampler.run_mcmc(initial_position, nsteps, progress=True, skip_initial_state_check=True)

# samples = sampler.get_chain(flat=True)
samples = sampler.get_chain()
# print(chain.shape)

np.savez(
    "chains_emcee.npz",
    chains=samples,
)
labels = ["$M_c$", "$\eta$", "$\chi_1$", "$\chi_2$", "D", "$t_c$", "$\phi_c$"]

fig = corner.corner(samples, labels=labels, truths=[Mc, eta, 0.3, -0.4, distance, 0, 0. ])
plt.savefig("emcee_test.pdf", bbox_inches="tight")

# n_dim = 7
# n_chains = 4
# n_local_steps = 1500000
# n_global_steps = 50000
# step_size = 0.01
# n_loop_training = 5
# n_loop_production = 1
# n_leapfrog = 10

# true_params = jnp.array([Mc, eta, 0.3, -0.4])
# rng_key_set = initialize_rng_keys(n_chains, seed=41)

# initial_noise = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
# initial_mean = jnp.array([Mc, eta, 0.3, -0.4, distance, 0, 0. ]).reshape(1,7)
# sigma = jnp.array([1, 0.001, 0.1, 0.1, 10, 0.001, 0.01]).reshape(1,7)

# initial_position = np.array(initial_mean + sigma * initial_noise)
# initial_position[initial_position[:,1]>0.25,1] = 0.25
# initial_position[initial_position[:,6]<0.,6] = 0.0
# initial_position = jnp.array(initial_position)

# mass_diag = lambda x: jnp.abs(1./(jax.grad(logL)(x)+jax.grad(top_hat)(x)))

# mass_matrix = jnp.array(np.ones(n_dim))
# mass_matrix = mass_matrix.at[1].set(mass_matrix[1]/300)
# mass_matrix = mass_matrix.at[4].set(mass_matrix[4]*100)
# mass_matrix = mass_matrix.at[5].set(mass_matrix[5]/100)
# mass_matrix = mass_matrix.at[6].set(mass_matrix[6]/10)



# local_sampler = GaussianRandomWalk(
#     posterior,
#     True,
#     {
#         "step_size": step_size*mass_matrix,
#     },
# )

# model = RQSpline(n_dim, 4, [32, 32], 8)

# print("Initializing sampler class")

# nf_sampler = Sampler(
#     n_dim,
#     rng_key_set,
#     local_sampler,
#     posterior,
#     model,
#     n_loop_training=n_loop_training,
#     n_loop_production=n_loop_production,
#     n_local_steps=n_local_steps,
#     n_global_steps=n_global_steps,
#     n_chains=n_chains,
#     use_global=False,
# )

# nf_sampler.sample(initial_position)

# chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()

# np.savez(
#     "chains_gaussian.npz",
#     chains=chains,
#     log_prob=log_prob,
#     local_accs=local_accs,
#     global_accs=global_accs,
# )