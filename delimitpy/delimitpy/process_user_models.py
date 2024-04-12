import os
import numpy as np
import logging
import msprime
import configparser
import io
import pandas as pd
import ast
import demesdraw # ModelBuilder
import matplotlib.pyplot as plt # ModelBuilder
from matplotlib.backends.backend_pdf import PdfPages
import copy

class ModelReader:

    """Read user specified models."""

    def __init__(self, config_values):
        self.config = config_values
        self.rng = np.random.default_rng(self.config['seed'])

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def read_models(self):

        # get list of .model files
        model_files = [x for x in os.listdir(self.config["user models"]) if x.endswith('model')]

        # iterate over model files
        demographies = []
        labels = []
        for model in model_files:

            modelinfo = configparser.ConfigParser(inline_comment_prefixes="#")
            modelinfo.optionxform = str
            modelinfo.read(os.path.join(self.config["user models"],model))

            # iterate over parameterizations
            for _ in range(self.config["replicates"]):

                # empty demography
                demography = msprime.Demography()

                # iterate over populations and add to demogrpahy
                for item in modelinfo["Populations"]:
                    size_range = [float(val.strip("[").strip("]")) \
                    for val in modelinfo['Populations'][item].split(",")]
                    initial_size = np.round(self.rng.uniform(low=size_range[0], high=size_range[1], size=1),0)[0]
                    demography.add_population(name = item, initial_size=initial_size)

                for item in modelinfo["Migration"]:
                    migration_df = pd.read_csv(io.StringIO(modelinfo["Migration"][item]), sep="\t")
                    for index, row in migration_df.iterrows():
                        for colname, value in row.items():
                            if index != colname and value != 0 and value != "0":
                                migrationrate_range = [float(val.strip("[").strip("]")) \
                                    for val in value.split(",")]
                                migration_rate = self.rng.uniform(low=migrationrate_range[0], high=migrationrate_range[1], size=1)
                                demography.set_migration_rate(source=index, dest=colname, rate=migration_rate)

                for item in modelinfo["Events"]:

                    infolist = modelinfo["Events"][item].split("\t")

                    # get type of event
                    type = infolist[0]

                    if type == 'split':
                        derived = ast.literal_eval(infolist[3])
                        ancestral = infolist[4]
                        mindiv = int(infolist[1])
                        maxdiv = int(infolist[2])
                        divergence_time = np.round(self.rng.uniform(low=mindiv, high=maxdiv, size=1),0)[0]
                        demography.add_population_split(derived=derived, ancestral=ancestral, time=divergence_time)

                    elif type == 'symmetric migration':
                        populations = ast.literal_eval(infolist[3])
                        mintime = int(infolist[1])
                        maxtime = int(infolist[2])
                        migration_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        try:
                            migration_rate = float(infolist[4])
                        except:
                            migration_rate_range = ast.literal_eval(infolist[4])
                            migration_rate = self.rng.uniform(low=migration_rate_range[0], high=migration_rate_range[1], size=1)[0]
                        demography.add_symmetric_migration_rate_change(populations=populations, time=migration_time, rate=migration_rate)

                    elif type == 'asymmetric migration':
                        source=infolist[3]
                        dest=infolist[4]
                        mintime = int(infolist[1])
                        maxtime = int(infolist[2])
                        migration_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        try:
                            migration_rate = float(infolist[5])
                        except:
                            migration_rate_range = ast.literal_eval(infolist[5])
                            migration_rate = self.rng.uniform(low=migration_rate_range[0], high=migration_rate_range[1], size=1)[0]
                        demography.add_migration_rate_change(source=source, dest=dest, time=migration_time, rate=migration_rate)

                    elif type == 'popsize':
                        mintime = int(infolist[1])
                        maxtime = int(infolist[2])
                        pop_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        population = infolist[3]
                        if infolist[4] != 'None':
                            population_size_range = ast.literal_eval(infolist[4])
                            population_size = np.round(self.rng.uniform(low=population_size_range[0], high=population_size_range[1], size=1),0)[0]
                        else:
                            population_size = None
                        if infolist[5] != 'None':
                            growth_rate_range = ast.literal_eval(infolist[5])
                            growth_rate = self.rng.uniform(low=growth_rate_range[0], high=growth_rate_range[1], size=1)[0]
                        else:
                            growth_rate = None

                        demography.add_population_parameters_change(population=population, time=pop_time, initial_size=population_size, growth_rate=growth_rate)

                    elif type == 'bottleneck':
                        mintime = int(infolist[1])
                        maxtime = int(infolist[2])
                        pop_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        population = infolist[3]
                        bottleneck_prop_range = ast.literal_eval(infolist[4])
                        bottleneck_prop = self.rng.uniform(low=bottleneck_prop_range[0], high=bottleneck_prop_range[1], size=1)[0]
                        demography.add_simple_bottleneck(population=population, time=pop_time, proportion=bottleneck_prop)

                    else:
                        raise Exception(f"Type {type} is not a valid option. Valid options include split, symmetric migration, asymmetric migration, popsize, and bottleneck.")






                demography.sort_events()
                demographies.append(demography)
                labels.append(model.split(".model")[0])
        return(demographies, labels)

    def validate_models(self, demographies, labels, outplot=None):
        """
        Plot example models demographies.

        Parameters:
            demographies (List): demographies
            labels (List): model labels
            outplot (string): path to store output figures. Default is to show.

        Returns:
            Nothing

        Raises:
            Error: If models cannot be plotted.
        """

        try:
            # Plot divergence demographies
            self._plot_models(demographies, labels, outplot)


        except ValueError as ve:
            raise ValueError(f"ValueError: Issue when plotting example \
                             msprime demographies: {ve}") from ve

        except Exception as e:
            raise RuntimeError(f"Unexpected Error: Issue when plotting \
                               example msprime demographies: {e}") from e


    def _nonzero(self, model):
        count=0
        for event in model.events:
            if event.time == 0:
                count+=1
                event.time = count*1
        return model

    def _plot_models(self, demographies, labels, outplot):
        """Plot example models for a given type of demography."""

        if outplot is None:
            for modelix, model in enumerate(demographies):
                if modelix % self.config['replicates'] == 0:
                    new_model = copy.deepcopy(model)
                    new_model = self._nonzero(new_model)
                    graph = new_model.to_demes()

                    # Plot the model
                    fig = plt.subplots()
                    demesdraw.tubes(graph, ax=fig[1], seed=1)
                    plt.title(f"Model: {labels[modelix]}")
                    plt.show()

        else:
            with PdfPages(outplot) as pdf:
                for modelix, model in enumerate(demographies):
                    if modelix % self.config['replicates'] == 0:
                        new_model = copy.deepcopy(model)
                        new_model = self._nonzero(new_model)
                        graph = new_model.to_demes()

                        # Plot the model
                        fig, ax = plt.subplots()
                        demesdraw.tubes(graph, ax=ax, seed=1)
                        plt.title(f"Model: {labels[modelix]}")
                        pdf.savefig(fig)
                        plt.close(fig)
