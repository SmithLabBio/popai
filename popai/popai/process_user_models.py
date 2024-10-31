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
import re

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
        model_files_unsorted = [x for x in os.listdir(self.config["user models"]) if x.endswith('model')]
        model_files = sorted(model_files_unsorted, key=lambda x: int(x.split('_')[1].split('.')[0]))


        # iterate over model files
        demographies = []
        labels = []
        for model in model_files:

            # list for storing active populations and populations with migration at the present.
            active_populations = []
            pops_w_present_migration = []

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
                    active_populations.append(item)

                for item in modelinfo["Migration"]:
                    migration_df = pd.read_csv(io.StringIO(modelinfo["Migration"][item]), sep="\t")
                    for index, row in migration_df.iterrows():
                        for colname, value in row.items():
                            if index != colname and value != 0 and value != "0":
                                migrationrate_range = [float(val.strip("[").strip("]")) \
                                    for val in value.split(",")]
                                migration_rate = self.rng.uniform(low=migrationrate_range[0], high=migrationrate_range[1], size=1)
                                demography.set_migration_rate(source=index, dest=colname, rate=migration_rate)
                                pops_w_present_migration.append(index)
                                pops_w_present_migration.append(colname)

                # dictionary for storing event names and times
                event_dict = {}

                for item in modelinfo["Events"]:

                    infolist = modelinfo["Events"][item].split("\t")

                    # get type of event
                    type = infolist[0]

                    if type == 'split':
                        derived = ast.literal_eval(infolist[3])
                        ancestral = infolist[4]
                        try:
                            mindiv = int(infolist[1])
                        except:
                            mindiv = infolist[1]
                            mindiv_str = self._replace_variables(mindiv, event_dict)
                            mindiv = eval(f"f'{mindiv_str}'")
                            mindiv = eval(mindiv)

                        
                        try:
                            maxdiv = int(infolist[2])
                        except:
                            maxdiv = infolist[2]
                            maxdiv_str = self._replace_variables(maxdiv, event_dict)
                            maxdiv = eval(f"f'{maxdiv_str}'")
                            maxdiv = eval(maxdiv)

                        divergence_time = np.round(self.rng.uniform(low=mindiv, high=maxdiv, size=1),0)[0]
                        if len(infolist) == 6:
                            event_dict[infolist[5]] = divergence_time
                        demography.add_population_split(derived=derived, ancestral=ancestral, time=divergence_time)
                        active_populations = [x for x in active_populations if x != ancestral]


                    elif type == 'symmetric migration':
                        populations = ast.literal_eval(infolist[3])
                        try:
                            mintime = int(infolist[1])
                        except:
                            mintime = infolist[1]
                            mintime_str = self._replace_variables(mintime, event_dict)
                            mintime = eval(f"f'{mintime_str}'")
                            mintime = eval(mintime)
                        try:
                            maxtime = int(infolist[2])
                        except:
                            maxtime = infolist[2]
                            maxtime_str = self._replace_variables(maxtime, event_dict)
                            maxtime = eval(f"f'{maxtime_str}'")
                            maxtime = eval(maxtime)

                        migration_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        if len(infolist) == 6:
                            event_dict[infolist[5]] = migration_time
                        try:
                            migration_rate = float(infolist[4])
                        except:
                            migration_rate_range = ast.literal_eval(infolist[4])
                            migration_rate = self.rng.uniform(low=migration_rate_range[0], high=migration_rate_range[1], size=1)[0]
                        demography.add_symmetric_migration_rate_change(populations=populations, time=migration_time, rate=migration_rate)

                    elif type == 'asymmetric migration':
                        source=infolist[3]
                        dest=infolist[4]
                        try:
                            mintime = int(infolist[1])
                        except:
                            mintime = infolist[1]
                            mintime_str = self._replace_variables(mintime, event_dict)
                            mintime = eval(f"f'{mintime_str}'")
                            mintime = eval(mintime)
                        try:
                            maxtime = int(infolist[2])
                        except:
                            maxtime = infolist[2]
                            maxtime_str = self._replace_variables(maxtime, event_dict)
                            maxtime = eval(f"f'{maxtime_str}'")
                            maxtime = eval(maxtime)
                        migration_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        if len(infolist) == 7:
                            event_dict[infolist[6]] = migration_time
                        try:
                            migration_rate = float(infolist[5])
                        except:
                            migration_rate_range = ast.literal_eval(infolist[5])
                            migration_rate = self.rng.uniform(low=migration_rate_range[0], high=migration_rate_range[1], size=1)[0]
                        demography.add_migration_rate_change(source=source, dest=dest, time=migration_time, rate=migration_rate)

                    elif type == 'popsize':
                        try:
                            mintime = int(infolist[1])
                        except:
                            mintime = infolist[1]
                            mintime_str = self._replace_variables(mintime, event_dict)
                            mintime = eval(f"f'{mintime_str}'")
                            mintime = eval(mintime)
                        try:
                            maxtime = int(infolist[2])
                        except:
                            maxtime = infolist[2]
                            maxtime_str = self._replace_variables(maxtime, event_dict)
                            maxtime = eval(f"f'{maxtime_str}'")
                            maxtime = eval(maxtime)
                        pop_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        if len(infolist) == 7:
                            event_dict[infolist[6]] = pop_time
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
                        try:
                            mintime = int(infolist[1])
                        except:
                            mintime = infolist[1]
                            mintime_str = self._replace_variables(mintime, event_dict)
                            mintime = eval(f"f'{mintime_str}'")
                            mintime = eval(mintime)
                        try:
                            maxtime = int(infolist[2])
                        except:
                            maxtime = infolist[2]
                            maxtime_str = self._replace_variables(maxtime, event_dict)
                            maxtime = eval(f"f'{maxtime_str}'")
                            maxtime = eval(maxtime)

                        pop_time = np.round(self.rng.uniform(low=mintime, high=maxtime, size=1),0)[0]
                        if len(infolist) == 6:
                            event_dict[infolist[5]] = pop_time
                        population = infolist[3]
                        bottleneck_prop_range = ast.literal_eval(infolist[4])
                        bottleneck_prop = self.rng.uniform(low=bottleneck_prop_range[0], high=bottleneck_prop_range[1], size=1)[0]
                        demography.add_simple_bottleneck(population=population, time=pop_time, proportion=bottleneck_prop)

                    else:
                        raise Exception(f"Type {type} is not a valid option. Valid options include split, symmetric migration, asymmetric migration, popsize, and bottleneck.")



                # raise warning if migration matrix includes migration between populations not extant at the present.
                missing_pops = set(pops_w_present_migration) - set(active_populations)
                if missing_pops:
                    logging.warning(f"The migration matrix for model {model} includes migration between populations not extant at the present. We recommend using events for such migration, rather than the matrix. This prevent issues with plotting.")

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
                    try:
                        graph = new_model.to_demes()

                        # Plot the model
                        fig = plt.subplots()
                        demesdraw.tubes(graph, ax=fig[1], seed=1)
                    except:
                        logging.warning(f"There was an issue with plotting your model. Please verify that your model is working as desired. This could arise if the migration matrix includes migration between populations not extant at the present. We recommend using events for such migration, rather than the matrix. This prevent issues with plotting.")
                    plt.title(f"Model: {labels[modelix]}")
                    plt.show()

        else:
            with PdfPages(outplot) as pdf:
                for modelix, model in enumerate(demographies):
                    if modelix % self.config['replicates'] == 0:
                        new_model = copy.deepcopy(model)
                        new_model = self._nonzero(new_model)
                        try:
                            graph = new_model.to_demes()

                            # Plot the model
                            fig, ax = plt.subplots()
                            demesdraw.tubes(graph, ax=ax, seed=1)
                        except:
                            # Clear the plot
                            ax.clear()

                            # Add the warning text in the center of the blank page
                            ax.text(0.5, 0.5, "There was an issue with plotting your model. Please verify that your model is working as desired.\nThis could arise if the migration matrix includes migration between populations not extant at the present.\nWe recommend using events for such migration, rather than the matrix. This will prevent issues with plotting.",
                                    ha='center', va='center', wrap=True, fontsize=12)

                            # Hide axes
                            ax.set_xticks([])
                            ax.set_yticks([])
                            logging.warning(f"There was an issue with plotting your model. Please verify that your model is working as desired. This could arise if the migration matrix includes migration between populations not extant at the present. We recommend using events for such migration, rather than the matrix. This prevent issues with plotting.")

                        plt.title(f"Model: {labels[modelix]}")
                        pdf.savefig(fig)
                        plt.close(fig)

    def _replace_variables(self, expr, event_dict):
        operators = ['min','max', 'mean', 'median']
        def replacer(match):
            var = match.group(0)
            return f'{{{event_dict[var]}}}' if var not in operators else var
    
        # Use regex to find words
        return re.sub(r'([A-Za-z_]\w*)', replacer, expr)
