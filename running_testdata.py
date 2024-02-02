from delimitpy import generate_test_data
import msprime

# simulate a test dataset

# build the demgoraphy
sample_sizes = {"A":6, "B":4, "C":5}
ne = [20000,20000,20000,20000,20000]
tdiv1 = 25000
tdiv2 = 250000
my_test_demography = msprime.Demography()
count=0
for key in sample_sizes:
    my_test_demography.add_population(name=key, initial_size=ne[count])
    count+=1
my_test_demography.add_population(name='AB', initial_size=ne[3])
my_test_demography.add_population(name='ABC', initial_size=ne[4])
my_test_demography.add_population_split(derived=["A","B"], ancestral="AB", time=tdiv1)
my_test_demography.add_population_split(derived=["AB","C"], ancestral="ABC", time=tdiv2)

# simulate the data
test_data_generator = generate_test_data.TestDataGenerator(sampling_dictionary=sample_sizes, fragments = 10, min_length = 1000, max_length = 3000, seed = 1234, substitution_model="jc69", model=my_test_demography, outdir="test_dataset", mutation_rate=5e-8)
test_data_generator.simulate()
