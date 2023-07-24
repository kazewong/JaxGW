from abc import ABC, abstractmethod
import jax.numpy as jnp
import numpy as np
import h5py
import json # to read JSON file
import requests # to download data file via URL
import os.path

# It bookmarks functions that fetch the data, store the data, and format the data
class PosteriorSampleData:
    # Constructor
    def __init__(self, data_file):
        self.posterior_samples = None
        self.data_file = data_file
        pass
    
    # download the .h5 posterior samples data into a directory
    def fetch(self, directory = "data/"):
        self.data_file = directory
        # opening JSON file
        event_list_file = open("event_list.json", "r")
        # return "files" element in JSON object as a dictionary
        event_list = (json.load(event_list_file))['files']
        # Download the posterior samples via url for each event listed in event_list.json
        for event in event_list[:5]:
            if event['type'] == 'h5': # Check if the event links to a H5 file
                if (event['key'][-14:]) == 'mixed_cosmo.h5': # We only want cosmological reweighted data
                    url = event['links']['self'] # get the url
                    filename = event['key'] # get the file name
                    
                    if os.path.isfile(directory + filename) == False: # if the file does not exist
                        print('Downloading ' + filename)
                        r = requests.get(url, allow_redirects=True)
                        open(directory + filename, 'wb').write(r.content) # download the data file into the data folder
                    else: # if the already exist
                        print(filename + ' exists')
    
    def read_file(self, directory=None, data_type = "C01:Mixed"):
        if (directory == None):
            directory = self.data_file
            if (directory == None):
                print("No data file directory specified. ")
                return None
        posterior_samples = []
        for file in os.listdir(directory): # loop through files in the data folder
            posterior_samples.append(h5py.File(directory+file)[data_type+"/posterior_samples"]) # append the address of dataframe to the list
        self.posterior_samples = posterior_samples
    
    # Read the .h5 data from a data folder and copy them into a python list
    def get_all_posterior_samples(self, directory = None):
        # If there is no posterior samples stored in this object, get the data from the .h5 file first
        if self.posterior_samples == None:
            self.read_file(directory)
        return self.posterior_samples
        
    
    # Get all the posterior samples of one specific parameters 
    def get_posterior_samples(self, params):
        if (self.posterior_samples == None):
            self.read_file()
        return [[events[param] for param in params] for events in self.posterior_samples]
        
        
    
        
        
# It stores the evaluation methods for calculating population model (probability of population parameters given posterior samples)
class PopulationModelBase:
    # Constructor
    def __init__(self) -> None:
        self.population_params_list = None
        
    @abstractmethod
    def get_population_params_list(self):
        return self.population_params_list
    
    @abstractmethod
    def get_population_likelihood(self):
        pass
    
    @abstractmethod
    def get_population_prior(self):
        pass
    
    def log_uniform_prior(self, min, max, x):
        return jnp.where((x < min) | (x > max), -np.infty, 0.0)

    
class PowerLawModel(PopulationModelBase):
    # Constructor
    def __init__(self):
        self.population_params_list = ["mass_1_source", "mass_ratio"]
    
    def get_population_params_list(self):
        return self.population_params_list
    
    # Evaluate population likelihood by power law
    def get_population_likelihood(self, population_params, posterior_samples):
        alpha, beta, m_min, m_max = population_params[0], population_params[1], population_params[2], population_params[3]
        m_1, q = posterior_samples[0], posterior_samples[1]
        epsilon = 0.001 # a very small number for limit computation

        normalization_constant = 1.0
        normalization_constant *= jnp.where((alpha>(1.0-epsilon))&(alpha<(1.0+epsilon)), jnp.log(m_max/m_min), (m_max**(1.0-alpha)-m_min**(1.0-alpha))/(1.0-alpha))
        
        if (beta > (-1.0-epsilon)) & (beta<(-1.0+epsilon)):
            return 0.0 # The normalization constant will be negative infinity, this gives 0
        else:
            normalization_constant *= (1.0 / (beta + 1.0))
        
        return jnp.where((m_1 > m_min) & (m_1 < m_max),
                        (m_1 ** (-alpha)) * (q ** beta) / normalization_constant,
                        0.0)

    # Evaluate the prior of the power law
    def get_population_prior(self, population_params):
        alpha, beta, m_min, m_max = population_params[0], population_params[1], population_params[2], population_params[3] # alpha, beta,... are double
        output = super().log_uniform_prior(-4., 12., alpha) + super().log_uniform_prior(-4., 12., beta) + super().log_uniform_prior(2.,10.,m_min) + super().log_uniform_prior(30.,100.,m_max)
        return output





# It evaluates the probability of population parameters given data
class PopulationDistribution:
    def __init__(self, model, sample_data):
        self.model = model
        self.sample_data = sample_data
        self.posterior_sample = self.sample_data.get_posterior_samples(self.model.get_population_params_list())
        
    def get_distribution(self, population_params, data):
        # check on population parameters
        population_prior = self.model.get_population_prior(population_params)
        
        # if parameters are ok, do the computation
        log_population_distribution = 0.0 # initialize the value to zero
        for event in self.posterior_sample:
            sum = np.sum(self.model.get_population_likelihood(population_params, event))
            log_population_distribution += (population_prior + np.log(sum) - np.log(event.shape[0])) # sum divided by the number of samples                     
        
        if np.isfinite(log_population_distribution):
            return log_population_distribution
        else:
            return -np.inf
        

# Import some necessary modules
import jax
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.realNVP import RealNVP
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler

data = PosteriorSampleData('data/')
m = PowerLawModel()
p = PopulationDistribution(model=PowerLawModel(), sample_data=data)

n_dim = 4
n_chains = 20
rng_key_set = initialize_rng_keys(n_chains, seed=42)
initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
param_initial_guess = [2.5, 6.0, 4.5, 80.0]
for i, param in enumerate(param_initial_guess):
    initial_position = initial_position.at[:, i].add(param)

n_layer = 10  # number of coupling layers
n_hidden = 128  # with of hidden layers in MLPs parametrizing coupling layers
model = RealNVP(n_layer, n_dim, n_hidden, jax.random.PRNGKey(21))

step_size = 1e-1
MALA_Sampler = MALA(p.get_distribution, True, {"step_size": step_size})
local_sampler_caller = lambda x: MALA_Sampler.make_sampler()



n_loop_training = 20
n_loop_production = 100
n_local_steps = 100
n_global_steps = 10
num_epochs = 5

learning_rate = 0.005
momentum = 0.9
batch_size = 5000
max_samples = 5000

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    None,
    MALA_Sampler,
    # target_dual_moon,
    model,
    n_loop_training=n_loop_training,
    n_loop_production=n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    use_global=True,
)

nf_sampler.sample(initial_position, data=None)

