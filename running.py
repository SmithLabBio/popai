from delimitpy import parse_input, generate_models, simulate_data, generate_test_data, build_predictors
import os
import numpy as np

# Parse the configuration file
config_parser = parse_input.ModelConfigParser("config_local.txt")
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



# Simulate data
data_simulator = simulate_data.DataSimulator(parameterized_models, labels, config=config_values)
simulated_ancestries = data_simulator.simulate_ancestry()
simulated_mutations = data_simulator.simulate_mutations()
arrays = data_simulator.mutations_to_numpy()
sfs = data_simulator.mutations_to_sfs()
stats = data_simulator.mutations_to_stats()
np.save(os.path.join(config_values["output directory"], 'labels.npy'), np.array(labels), allow_pickle=True)
np.save(os.path.join(config_values["output directory"], 'numpy_arrays.npy'), np.array(arrays), allow_pickle=True)
np.save(os.path.join(config_values["output directory"], 'sfs.npy'), np.array(sfs), allow_pickle=True)
np.save(os.path.join(config_values["output directory"], 'stats.npy'), np.array(stats), allow_pickle=True)

arrays = np.load(os.path.join(config_values["output directory"], 'numpy_arrays.npy'))
sfs = np.load(os.path.join(config_values["output directory"], 'sfs.npy'))
stats = np.load(os.path.join(config_values["output directory"], 'stats.npy'))

# Fit models
random_forest_sfs_predictor = build_predictors.RandomForests_SFS(config_values, sfs, labels)
random_forest_sfs_predictor.build_rf_sfs()