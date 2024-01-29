from delimitpy import generate_models, simulate_data

# Parse the configuration file
config_parser = generate_models.ModelConfigParser("config_local.txt")
config_values = config_parser.parse_config()

# build the models
model_builder = generate_models.ModelBuilder(config_values)
model_builder.build_models()

# parameterize the models
parameterized_models, labels = model_builder.draw_parameters()

## validate the models
#model_builder.validate_models(parameterized_models, labels)

## write models to yaml: still needs work
#model_writer = generate_models.ModelWriter(config_values, parameterized_divergence_demographies, parameterized_sc_demographies, parameterized_dwg_demographies)
#model_writer.write_to_yaml()

# read models from yaml: still needs work
#model_reader = generate_models.ModelReader('./test', 10, seed=1234)
#model_reader.read_yaml()

sample_sizes = {"A":12, "B":10, "C":14}
data_simulator = simulate_data.DataSimulator(parameterized_models, labels, config=config_values, sample_sizes=sample_sizes)
simulated_ancestries = data_simulator.simulate_ancestry()
simulated_mutations = data_simulator.simulate_mutations()