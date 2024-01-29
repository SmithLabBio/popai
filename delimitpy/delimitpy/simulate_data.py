"""This module contains all Classes for simulating datasets under specified models using msprime."""
import logging
logging.getLogger('msprime').setLevel("WARNING")
import msprime
import numpy as np
import dendropy

class DataSimulator:

    """Simulate data under specified demographies."""

    def __init__(self, models, labels, config, sample_sizes):
        self.models = models
        self.labels = labels
        self.config = config
        self.sample_sizes = sample_sizes
        

        self.rng = np.random.default_rng(self.config['seed'])
    
    def simulate_ancestry(self):



        ts_ancestry = []

        for demography in self.models:

            ancestry_seeds = self.rng.integers(2**32, size=len(demography))

            # figure out how to sample individuals
            sampling_dict = {}
            populations = [x.name for x in demography[0].populations]
            sampling_dict = {population: 0 for population in populations}

            for species in self.config['species tree'].leaf_nodes():
                if species.taxon.label not in populations:
                    search = True
                    searchnode = species
                    while search == True:
                        if searchnode.parent_node.label in populations:
                            sampling_dict[searchnode.parent_node.label]+=self.sample_sizes[species.taxon.label]
                            search = False
                        else:
                            searchnode = searchnode.parent_node
                else:
                    sampling_dict[species.taxon.label] += self.sample_sizes[species.taxon.label]
            sampling_dict = {key: value for key, value in sampling_dict.items() if value != 0}

            model_ts = []

            for parameterization in enumerate(demography):
                # simulate data
                ts = msprime.sim_ancestry(sampling_dict, demography=parameterization[1], random_seed = ancestry_seeds[parameterization[0]])
                model_ts.append(ts)
            
            ts_ancestry.append(model_ts)
        
        self.ts_ancestry = ts_ancestry
        return(ts_ancestry)

    def simulate_mutations(self):

        mts_mutations = []

        for model in self.ts_ancestry:

            mutation_rates = self.rng.uniform(low=self.config["mutation_rate"][0], high=self.config["mutation_rate"][1], size=self.config["replicates"])

            model_mts = []

            for replicate in enumerate(model):

                mts = msprime.sim_mutations(replicate[1], rate=mutation_rates[replicate[0]], model=self.config["substitution model"])
                model_mts.append(mts)

            mts_mutations.append(model_mts)
            
        return(mts_mutations)