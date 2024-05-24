from popai import generate_test_data
import msprime

# simulate a test dataset

# build the demgoraphy
sample_sizes = {"A":10, "B":10, "C":10}
ne = [20000,20000,20000,20000,20000]
tdiv1 = 25000
tdiv2 = 750000
my_test_demography = msprime.Demography()
count=0
for key in sample_sizes:
    my_test_demography.add_population(name=key, initial_size=ne[count])
    count+=1
my_test_demography.set_symmetric_migration_rate(populations=["A","B"], rate=5e-5)
my_test_demography.add_population(name='AB', initial_size=ne[3])
my_test_demography.add_population(name='ABC', initial_size=ne[4])
my_test_demography.add_population_split(derived=["A","B"], ancestral="AB", time=tdiv1)
my_test_demography.add_population_split(derived=["AB","C"], ancestral="ABC", time=tdiv2)

# simulate the data
test_data_generator = generate_test_data.TestDataGenerator(sampling_dictionary=sample_sizes, fragments = 20, min_length = 3000, max_length = 4000, seed = 1234, 
                                                           substitution_model="jc69", model=my_test_demography, outdir="./alignments", mutation_rate=1e-8)
test_data_generator.simulate(missing=0.0, missing_ind=0.0, format=["fasta","vcf"])
