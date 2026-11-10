"""This file contains the Classes and Functions for simulating a test dataset."""
import msprime
import os
import pyslim
import numpy as np
import dendropy
import sys

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
        # only JC69, as in parse_input: msprime's HKY and GTR need parameters (kappa, relative
        # rates), which popai can't take yet (see the to-do list)
        if substitution_model.lower() != "jc69":
            raise ValueError(f"Substitution model must be JC69, not {substitution_model}.")
        self.mutation_model = msprime.JC69()
        self.rng = np.random.default_rng(self.seed)
    
    def simulate(self, missing=0.0, missing_ind=0.0, format=["fasta"]):

        # get seeds and lengths
        ancestry_seeds = self.rng.integers(2**32, size=self.fragments)
        if self.max_length == self.min_length:
            lengths = [self.min_length] * self.fragments
        else:
            lengths = self.rng.integers(low=self.min_length, high=self.max_length, size=self.fragments)

        # set up vcf header
        vcf_header = None
        vcf_header_2 = None
        vcf_header_3 = None
        all_vcf_data = []

        for fragment in enumerate(lengths):
            
            # generate a reference
            reference_sequence = self.rng.choice(self.mutation_model.alleles, p=self.mutation_model.root_distribution, size=fragment[1])
            reference_sequence = ''.join(reference_sequence)
            
            ts = msprime.sim_ancestry(self.sampling_dictionary, demography=self.model, random_seed = ancestry_seeds[fragment[0]], sequence_length=fragment[1])
            mts = msprime.sim_mutations(ts, rate=self.mutation_rate, model=self.mutation_model, random_seed = ancestry_seeds[fragment[0]])
            if "fasta" in format:
                fasta_string = mts.as_fasta(reference_sequence=reference_sequence)
                fasta_dict = self._fasta_to_dict(fasta_string)

                if missing > 0:
                    for item,value in fasta_dict.items():
                        fasta_dict[item] = self._replace_with_N(value, missing)

                if missing_ind > 0:
                    new_fasta_dict = {}
                    for item,value in fasta_dict.items():
                        if self.rng.random() > missing_ind:
                            new_fasta_dict[item] = value
                    fasta_dict = new_fasta_dict

                self._write_fasta(fasta_dict, path = os.path.join(self.outdir, 'alignment_%s.fa' % str(fragment[0])))
            
            if "vcf" in format:

                samples = mts.samples()
                for sample_id, h in zip(samples, mts.haplotypes(samples=samples)):
                    pop = mts.node(sample_id).population
                    #print(mts.node(sample_id))
                    #print(f"Sample {sample_id:<2} ({ts.population(pop).metadata['name']:^5}): {h}")
                
                vcf_lines = mts.as_vcf().split("\n")
                
                if vcf_header == None:
                    vcf_header = vcf_lines[0:3]
                    vcf_header[1] = "##source=popai"
                    vcf_header_2 = [vcf_lines[3]]
                    vcf_header_3 = [vcf_lines[4]]
                    vcf_header_3.append('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">')
                    vcf_header_3.append(vcf_lines[5].replace('tsk_', 'n'))

                else:
                    vcf_header_2.append(f"##contig=<ID={fragment[0]+1},{vcf_lines[3].split(',')[1]}")

                vcf_data = vcf_lines[6:]
                for line in vcf_data:
                    if not line == "":
                        new_line = ""
                        for colix, genotype in enumerate(line.split()):
                            if colix == 0:
                                to_write=str(fragment[0]+1)
                            elif colix < 8:
                                to_write = genotype
                            elif colix == 8:
                                to_write = "GT:DP"
                            elif colix > 8:
                                if missing > 0 and self.rng.random() < missing:
                                    to_write = ".|.:0"
                                else:
                                    to_write = f"{genotype}:20"
                            new_line += to_write
                            new_line += "\t"
                        all_vcf_data.append(new_line)
                
                with open(os.path.join(self.outdir, 'alignment.vcf'), 'w') as f:
                    for line in vcf_header:
                        f.write(line)
                        f.write("\n")
                    for line in vcf_header_2:
                        f.write(line)
                        f.write("\n")
                    for line in vcf_header_3:
                        f.write(line)
                        f.write("\n")
                    for line in all_vcf_data:
                        f.write(line)
                        f.write("\n")


    def _fasta_to_dict(self, fasta_string):
        sequences = {}
        current_header = None
        for line in fasta_string.split('\n'):
            if line.startswith('>'):
                current_header = line[1:]
                sequences[current_header] = ''
            else:
                sequences[current_header] += line.strip()
        return(sequences)

    def _replace_with_N(self, string, proportion):
        if proportion < 0 or proportion > 1:
            raise ValueError("Proportion must be between 0 and 1")

        num_replacements = int(np.ceil(len(string) * proportion))
        indices_to_replace = self.rng.choice(len(string), size=num_replacements, replace=False)

        replaced_string = list(string)
        for index in indices_to_replace:
            replaced_string[index] = 'N'

        return ''.join(replaced_string)

    def _write_fasta(self, fasta_dict, path):
        with open(path, 'w') as f:
            for key,value in fasta_dict.items():
                f.write('>')
                f.write(key)
                f.write('\n')
                f.write(value)
                f.write('\n')



