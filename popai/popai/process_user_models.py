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
                    size_range = [float(val.strip("[").strip("]")) for val in modelinfo['Populations'][item].split(",")]
                    initial_size = np.round(self.rng.uniform(low=size_range[0], high=size_range[1], size=1),0)[0]
                    demography.add_population(name = item, initial_size=initial_size)
                    active_populations.append(item)

                # dictionary for storing event names and times
                event_dict = {}

                for item in modelinfo["Events"]:

                    item_dict = self._split_parameters(modelinfo["Events"][item])

                    if item_dict['event'] == 'split':

                        if not all(key in item_dict for key in ['time', 'descendants', 'ancestor']):
                            raise Exception(f"Check your split events. All must include a time, descendants, and an ancestor.")

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        demography.add_population_split(derived=item_dict['descendants'], ancestral=item_dict['ancestor'], time=event_time)
                        active_populations = [x for x in active_populations if x != item_dict['ancestor']]
                        event_dict[item] = event_time

                    elif item_dict['event'] == 'symmetric_migration':

                        if not all(key in item_dict for key in ['start', 'stop', 'populations', 'rate']):
                            raise Exception(f"Check your symmetric migration events. All must include a start and stop time, populations, and a rate.")

                        start_event_time = self._get_event_value(item_dict, event_dict, 'start')
                        stop_event_time = self._get_event_value(item_dict, event_dict, 'stop')
                        migration_rate = self._get_event_value(item_dict, event_dict, 'rate')

                        demography.add_symmetric_migration_rate_change(populations=item_dict['populations'], time=start_event_time, rate=migration_rate)
                        demography.add_symmetric_migration_rate_change(populations=item_dict['populations'], time=stop_event_time, rate=0)

                        event_dict[item] = [start_event_time, stop_event_time]

                    elif item_dict['event'] == 'asymmetric_migration':

                        if not all(key in item_dict for key in ['start', 'stop', 'source', 'dest', 'rate']):
                            raise Exception(f"Check your assymmetric migration events. All must include a start and stop time, a source, a dest, and a rate.")


                        start_event_time = self._get_event_value(item_dict, event_dict, 'start')
                        stop_event_time = self._get_event_value(item_dict, event_dict, 'stop')
                        migration_rate = self._get_event_value(item_dict, event_dict, 'rate')

                        demography.add_migration_rate_change(source=item_dict['source'], dest=item_dict['dest'], time=start_event_time, rate=migration_rate)
                        demography.add_migration_rate_change(source=item_dict['source'], dest=item_dict['dest'], time=stop_event_time, rate=0)

                        event_dict[item] = [start_event_time, stop_event_time]

                    elif item_dict['event'] == 'popsize':

                        if not all(key in item_dict for key in ['time', 'size', 'population']):
                            raise Exception(f"Check your assymmetric migration events. All must include a time, a population, and a size.")

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        population_size = self._get_event_value(item_dict, event_dict, 'size')

                        demography.add_population_parameters_change(population=item_dict['population'], time=event_time, initial_size=population_size, growth_rate=0)

                    elif item_dict['event'] == 'popgrowth':

                        if not all(key in item_dict for key in ['time', 'rate', 'population']):
                            raise Exception(f"Check your assymmetric migration events. All must include a time, a population, and a rate.")

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        growth_rate = self._get_event_value(item_dict, event_dict, 'rate')

                        demography.add_population_parameters_change(population=item_dict['population'], time=event_time, initial_size=None, growth_rate=growth_rate)

                        event_dict[item] = event_time

                    elif item_dict['event'] == 'bottleneck':

                        event_time = self._get_event_value(item_dict, event_dict, 'time')
                        bottleneck_prop = self._get_event_value(item_dict, event_dict, 'prop')

                        demography.add_simple_bottleneck(population=item_dict['population'], time=event_time, proportion=bottleneck_prop)
                        
                        event_dict[item] = event_time

                    else:
                        raise Exception(f"Type {item_dict['event']} is not a valid option. Valid options include split, symmetric migration, asymmetric migration, popsize, popgrowth, and bottleneck.")

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

    def _split_parameters(self, params):

        split_results = {}

        # get event type
        event_type, params = params.split('{')
        split_results['event'] = event_type

       # split parameters by commas, ignoring commas inside brackets.
        
        result = self._split_ignore_char(params, '[', ']')
        
        for item in result:
            if '[' in item.split('=')[1]:
                value = [x.strip('[] ') for x in self._split_ignore_char(item.split('=')[1], '(', ')')]
            else:
                value = item.split('=')[1].strip()
            split_results[item.split('=')[0].strip()] = value
        

        return split_results
    
    def _evaluate_var(self, var, event_dict, vartype):
        try:
            if vartype=='int':
                result = int(var)
            elif vartype=='float':
                result = float(var)
        except:
            result = var
            result_str = self._replace_variables(result, event_dict)
            result = eval(f"f'{result_str}'")
            result = eval(result)
        return(result)

    def _split_ignore_char(self, string, char1, char2):
        result = []
        current = []
        in_brackets = False

        for char in string:
            if char == char1:
                in_brackets = True
            elif char == char2:
                in_brackets = False
            elif char == "," and not in_brackets:
                result.append("".join(current).strip())
                current = []
                continue
            current.append(char)

        # Add the last parameter
        if current:
            result.append("".join(current).strip('}'))
        
        return(result)
    
    def _get_event_value(self, item_dict, event_dict, valuetype):
        
        
        if isinstance(item_dict[valuetype], list):
            
            if valuetype=='time' or valuetype=='start' or valuetype=='stop' or valuetype=='size':
                minval = self._evaluate_var(item_dict[valuetype][0], event_dict, 'int')
                maxval = self._evaluate_var(item_dict[valuetype][1], event_dict, 'int')
                event_value = self.rng.uniform(low=minval, high=maxval, size=1)[0]
                event_value = np.round(event_value,0)
            elif valuetype=='rate' or valuetype=='prop':
                minval = self._evaluate_var(item_dict[valuetype][0], event_dict, 'float')
                maxval = self._evaluate_var(item_dict[valuetype][1], event_dict, 'float')
                event_value = self.rng.uniform(low=minval, high=maxval, size=1)[0]
                
        
        else:
            try:
                event_value = item_dict[valuetype]
                if valuetype=='time' or valuetype=='start' or valuetype=='stop' or valuetype=='size':
                    event_value = int(event_value)
                elif valuetype=='rate' or valuetype=='prop':
                    event_value = float(event_value)
            except:
                event_value = self._evaluate_var(item_dict[valuetype], event_dict, 'float')
        
        return(event_value)
