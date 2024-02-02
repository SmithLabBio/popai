"""This file contains the Classes and Functions for simulating a test dataset."""
import numpy as np
import msprime
import os
import pyslim

class TestDataGenerator:

    """Parse user input from the configuration file."""

    def __init__(self, model, sampling_dictionary, substitution_model, fragments, min_length, max_length, seed, outdir, mutation_rate):
        self.model = model
        self.sampling_dictionary=sampling_dictionary
        self.substitution_model=substitution_model
        self.fragments=fragments
        self.min_length=min_length
        self.max_length=max_length
        self.outdir=outdir
        self.seed=seed
        self.mutation_rate = mutation_rate

        self.rng = np.random.default_rng(self.seed)
    
    def simulate(self):

        # get seeds and lengths
        ancestry_seeds = self.rng.integers(2**32, size=self.fragments)
        lengths = self.rng.integers(low=self.min_length, high=self.max_length, size=self.fragments)

        for fragment in enumerate(lengths):
            
            # generate a reference
            if self.substitution_model == 'jc69' or self.substitution_model == "JC69":
                mymutmodel = msprime.JC69()
            elif self.substitution_model == "hky" or self.substitution_model == "HKY":
                mymutmodel = msprime.HKY()
            elif self.substitution_model == "gtr" or self.substitution_model == "GTR":
                mymutmodel == msprime.GTR()
            else:
                raise("Error: Incorrect specification of substitution model.")
            reference_sequence = self.rng.choice(mymutmodel.alleles, p=mymutmodel.root_distribution, size=fragment[1])
            reference_sequence = ''.join(reference_sequence)
            
            ts = msprime.sim_ancestry(self.sampling_dictionary, demography=self.model, random_seed = ancestry_seeds[fragment[0]], sequence_length=fragment[1])
            mts = msprime.sim_mutations(ts, rate=self.mutation_rate, model=self.substitution_model)
            mts.write_fasta(file_or_path=os.path.join(self.outdir, 'alignment_%s.fa' % str(fragment[0])), reference_sequence=reference_sequence)






