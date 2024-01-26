import generate_models

# Parse the configuration file
config_parser = generate_models.ModelConfigParser("config.txt")
config_values = config_parser.parse_config()

# build the models
model_builder = generate_models.ModelBuilder(config_values)
model_builder.build_models()

# parameterize the models
parameterized_divergence_demographies, parameterized_sc_demographies, parameterized_dwg_demographies = model_builder.draw_parameters()

# validate the models
model_builder.validate_models(parameterized_divergence_demographies, parameterized_sc_demographies, parameterized_dwg_demographies)
