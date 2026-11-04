"""This module contains all Classes for creating msprime 
demographies to be used in downstream simulation."""

import copy # ModelBuilder
import pickle # ModelBuilder: fast copies of template demographies
from itertools import chain, combinations, product # ModelBuilder
import logging # ModelBuilder, ModelReader
import msprime # ModelBuilder
import numpy as np # ModelBuilder
import demesdraw # ModelBuilder
import matplotlib.pyplot as plt # ModelBuilder
from matplotlib.backends.backend_pdf import PdfPages

class ModelBuilder:

    """Generate a model set with parameters drawn from user-defined priors."""

    def __init__(self, config_values):
        self.config = config_values
        self.rng = np.random.default_rng(self.config['seed'])

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def build_models(self):
        """
        Build msprime models for divergence, secondary 
        contact, and divergence with gene flow scenarios.

        Parameters:
            config_values: the configuration info parsed using the ModelConfigParser module.

        Returns:
            a list of base msprime demographies

        Raises:
            Error: If demographies cannot be created.
        """
        try:

            all_divergence_demographies = []
            seen_divergence_models = set()
            full_model_trees = {}
            all_sc_demographies = []
            all_dwg_demographies = []
            total_models = 0

            for ix, tree in enumerate(self.config['species tree']):

                # get baseline divergence demographies
                divergence_demographies = self._create_divergence_demographies(tree)


                # warn if this tree is identical to an earlier one (the full model,
                # with no nodes collapsed, is the last one in the list)
                full_key = _divergence_model_key(divergence_demographies[-1])
                if full_key in full_model_trees:
                    self.logger.warning(
                        "Species trees %d and %d are identical (same topology and node "
                        "labels). Tree %d adds no new models, and its priors and migration "
                        "matrix are ignored.", full_model_trees[full_key] + 1, ix + 1, ix + 1)
                else:
                    full_model_trees[full_key] = ix

                to_keep = []
                for item in divergence_demographies:
                    key = _divergence_model_key(item)
                    if key not in seen_divergence_models:
                        seen_divergence_models.add(key)
                        to_keep.append(item)
                divergence_demographies = to_keep

                # get secondary contact demographies
                if self.config['secondary contact']:
                    sc_demographies = self._create_sc_demographies(divergence_demographies, self.config['migration df'][ix])
                else:
                    sc_demographies = []

                # get divergence with gene flow demographies
                if self.config['divergence with gene flow']:
                    dwg_demographies = self._create_dwg_demographies(divergence_demographies, self.config['migration df'][ix])
                else:
                    dwg_demographies = []

                total_models += len(divergence_demographies) + len(sc_demographies) \
                    + len(dwg_demographies)
                
                # add demographies to list
                all_divergence_demographies.append(divergence_demographies)
                all_sc_demographies.append(sc_demographies)
                all_dwg_demographies.append(dwg_demographies)
            
            self.logger.info("Creating %r different models for based on user input.", total_models)

        except ValueError as ve:
            raise ValueError(f"ValueError: Issue when building baseline"
                             f"msprime demographies: {ve}") from ve

        except Exception as e:
            raise RuntimeError(f"Error: Unexpected issue when building"
                               f"baseline msprime demographies: {e}") from e

        return(all_divergence_demographies, all_sc_demographies, all_dwg_demographies)

    def draw_parameters(self, divergence_demographies, sc_demographies, dwg_demographies):
        """
        Draw parameters for all models.

        Parameters:
            divergence_demographies (List): A list of divergence demography objects 
                returned from build_models.
            sc_demographies (List): A list of secondary contact demography objects 
                returned from build_models.
            dwg_demographies (List): A list of divergence with gene flow demography 
                objects returned from build_models.

        Returns:
            List: a list of demographies with parameters drawn from priors

        Raises:
            Error if priors are incorrectly defined.
        """
        all_parameterized_demographies = []
        labels = []
        sp_tree_index = []
        labelcount=0

        for ix, tree in enumerate(self.config['species tree']):

            # get priors
            population_sizes, divergence_times = _get_priors(tree)

            # draw parameters for divergence models
            if len(divergence_demographies[ix]) > 0:
                parameterized_divergence_demographies = self._draw_parameters_divergence(
                    divergence_times=divergence_times, population_sizes=population_sizes, \
                        divergence_demographies=divergence_demographies[ix])
                these_labels = [[iy]*len(x) for iy, x in enumerate(parameterized_divergence_demographies)]
                these_labels = [item for sublist in these_labels for item in sublist]
                these_labels = [x+labelcount for x in these_labels]
                labelcount=these_labels[-1]+1
                labels.extend(these_labels)
                these_sp_tree = [[ix] * len(these_labels)]
                sp_tree_index.extend(these_sp_tree)
                all_parameterized_demographies.extend(parameterized_divergence_demographies)

            # draw parameters for secondary contact models
            if len(sc_demographies[ix]) > 0:
                parameterized_sc_demographies = self._draw_parameters_sc(
                    divergence_times=divergence_times, population_sizes=population_sizes, \
                    sc_demographies=sc_demographies[ix])
                these_labels = [[iy]*len(x) for iy, x in enumerate(parameterized_sc_demographies)]
                these_labels = [item for sublist in these_labels for item in sublist]
                these_labels = [x+labelcount for x in these_labels]
                labelcount=these_labels[-1]+1
                labels.extend(these_labels)
                these_sp_tree = [[ix] * len(these_labels)]
                sp_tree_index.extend(these_sp_tree)
                all_parameterized_demographies.extend(parameterized_sc_demographies)

            # draw parameters for divergence with gene flow models
            if len(dwg_demographies[ix]) > 0:
                parameterized_dwg_demographies = self._draw_parameters_dwg(
                    divergence_times=divergence_times, population_sizes=population_sizes, \
                    dwg_demographies=dwg_demographies[ix])
                these_labels = [[iy]*len(x) for iy, x in enumerate(parameterized_dwg_demographies)]
                these_labels = [item for sublist in these_labels for item in sublist]
                these_labels = [x+labelcount for x in these_labels]
                labelcount=these_labels[-1]+1
                labels.extend(these_labels)
                these_sp_tree = [[ix] * len(these_labels)]
                sp_tree_index.extend(these_sp_tree)
                all_parameterized_demographies.extend([x for x in parameterized_dwg_demographies])
            
        # return them
        all_parameterized_demographies = [item for sublist in all_parameterized_demographies for item in sublist]
        sp_tree_index = [item for sublist in sp_tree_index for item in sublist]

        return(all_parameterized_demographies, labels, sp_tree_index)

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
            raise ValueError(f"ValueError: Issue when plotting example"
                             f"msprime demographies: {ve}") from ve

        except Exception as e:
            raise RuntimeError(f"Unexpected Error: Issue when plotting"
                               f"example msprime demographies: {e}") from e

    def _create_divergence_demographies(self, tree):
        """Create baseline divergence demographies."""

        demographies = []
        pop_size_holder = 1000
        div_time_holder = 1000

        # get all valid combos of nodes to collapse, then add the full model
        all_combos = self._collapse_combos(tree)
        all_combos.append([]) # add the full model

        # now generate the demographies for all combos
        for combo in all_combos:

            demography = msprime.Demography()

            # add populations that should be added, and set initial sizes (changed later)
            for internal_node in tree.postorder_internal_node_iter():
                if internal_node not in combo:
                    derived_1, derived_2 = self._get_derived_populations(internal_node)
                    demography.add_population(name=derived_1, initial_size=pop_size_holder)
                    demography.add_population(name=derived_2, initial_size=pop_size_holder)
            demography.add_population(name=str(tree.seed_node.label)\
                                      .strip("'"), initial_size=pop_size_holder)

            # add non-zero divergence events
            for internal_node in tree.postorder_internal_node_iter():
                if internal_node not in combo:
                    derived_1, derived_2 = self._get_derived_populations(internal_node)
                    ancestral = str(internal_node.label).strip("'")
                    demography.add_population_split(time=div_time_holder, derived=[derived_1,\
                         derived_2], ancestral=ancestral)

            demographies.append(demography)

        return demographies

    def _create_sc_demographies(self, divergence_demographies, migration_matrix):
        """Create baseline secondary contact demographies."""

        migration_demographies = []
        migtimeholder = 100
        migrateholder = 1e-3

        # iterate over demographies
        for item in divergence_demographies:

            # list of events to include
            to_include = sorted(set(self._find_sc_to_include(item, migration_matrix)))

            # keep only one per pair if symmetric rates are enforced
            if self.config['symmetric']:
                to_include_frozensets = []
                for pair in to_include:
                    sorted_pair = sorted(pair)
                    to_include_frozensets.append(tuple(sorted_pair))

                to_include = sorted(set(to_include_frozensets))

            # get all combos of events
            combos_of_migration = chain.from_iterable(combinations(to_include, r) for r in \
                range(1, min(self.config['max migration events'],len(to_include)) + 1))
            combos_of_migration = [list(combo) for combo in combos_of_migration]

            # create histories with each combo of events
            for combo in combos_of_migration:
                migration_demography = copy.deepcopy(item)

                for populationpair in combo:
                    populationpair=list(populationpair)

                    # if symmetric true, then add symmetric migration and add ceasing of migration
                    if self.config['symmetric']:
                        migration_demography.set_migration_rate(
                            source=populationpair[0], dest=populationpair[1], rate=migrateholder)
                        migration_demography.add_migration_rate_change(time=migtimeholder, \
                            source=populationpair[0], dest=populationpair[1], rate=0)
                        migration_demography.set_migration_rate(
                            source=populationpair[1], dest=populationpair[0], rate=migrateholder)
                        migration_demography.add_migration_rate_change(time=migtimeholder, \
                            source=populationpair[1], dest=populationpair[0], rate=0)

                    # if not symmetric then do asymmetric
                    else:
                        migration_demography.set_migration_rate(
                            source=populationpair[0], dest=populationpair[1], rate=migrateholder)
                        migration_demography.add_migration_rate_change(time=migtimeholder, \
                            source=populationpair[0], dest=populationpair[1], rate=0)

                # sort the events and add the demography to the list
                migration_demography.sort_events()
                migration_demographies.append(migration_demography)

        return migration_demographies

    def _create_dwg_demographies(self, divergence_demographies, migration_matrix):
        """Create baseline divergence with gene flow demographies."""

        migration_demographies = []
        migtimeholder = 100
        migrateholder = 1e-3

        # iterate over demographies
        for item in divergence_demographies:

            # list of events to include
            to_include = sorted(set(self._find_dwg_to_include(item, migration_matrix)))

            # keep only one per pair if symmetric rates are enforced
            if self.config['symmetric']:
                to_include_frozensets = []
                for pair in to_include:
                    sorted_pair = sorted(pair)
                    to_include_frozensets.append(tuple(sorted_pair))

                to_include = sorted(set(to_include_frozensets))

            # get all combos of events
            combos_of_migration = chain.from_iterable(combinations(to_include, r) for r in \
                range(1, min(self.config['max migration events'],len(to_include)) + 1))
            combos_of_migration = [list(combo) for combo in combos_of_migration]

            # create histories with each combo of events
            for combo in combos_of_migration:
                migration_demography = copy.deepcopy(item)

                for populationpair in combo:
                    populationpair=list(populationpair)
                    # if symmetric true, then add symmetric migration and add ceasing of migration
                    if self.config['symmetric']:
                        migration_demography.add_migration_rate_change(time=migtimeholder, \
                            source=populationpair[0], dest=populationpair[1], rate=migrateholder)
                        migration_demography.add_migration_rate_change(time=migtimeholder, \
                            source=populationpair[1], dest=populationpair[0], rate=migrateholder)
                    # if not symmetric then do asymmetric
                    else:
                        migration_demography.add_migration_rate_change(time=migtimeholder, \
                            source=populationpair[0], dest=populationpair[1], rate=migrateholder)

                # sort the events and add the demography to the list
                migration_demography.sort_events()
                migration_demographies.append(migration_demography)


        return migration_demographies

    def _collapse_combos(self, tree):
        """List every valid set of internal nodes to collapse, excluding the empty set.

        A set is valid when every collapsed node also has all of its internal
        descendants collapsed."""

        internal_nodes = tree.internal_nodes()
        position = {node: ix for ix, node in enumerate(internal_nodes)}

        def options(node):
            if node.is_leaf():
                return [[]]
            collapse_clade = [list(node.postorder_internal_node_iter())]
            keep_node = [[n for part in parts for n in part]
                         for parts in product(*(options(c) for c in node.child_nodes()))]
            return collapse_clade + keep_node

        combos = [sorted(c, key=position.get) for c in options(tree.seed_node) if c]
        combos.sort(key=lambda c: (len(c), [position[n] for n in c]))
        return combos

    def _get_derived_populations(self, internal_node):
        """Get the names of populations descending from an internal node."""

        children = internal_node.child_nodes()

        derived_1, derived_2 = [x.label if x.label is not None else x.taxon.label for x in children]

        return(derived_1, derived_2)

    def _find_sc_to_include(self, item, migration_matrix):
        """Find which secondary contact events we whould include for the demography, 'item'
        Include any events for which both populations exist before time zero and 
        there is migration allowed between the populations."""

        to_include = []

        # iterate over migration matrix and find populations with migration
        for index, row in migration_matrix.iterrows():
            for colname, value in row.items():
                include = [False,False]

                if index != colname and _allows_migration(value):

                    # check that the population diverged from its ancestor at a time > 0
                    for population in enumerate([index, colname]):
                        for event in item.events:
                            if population[1] in event.derived and event.time != 0:
                                include[population[0]] = True
                        for event in item.events:
                            if population[1] == event.ancestral and event.time != 0:
                                include[population[0]] = False

                # if both poulations meet are criteria, include the event.
                if include[0] and include[1]:
                    to_include.append((index,colname))

        return to_include

    def _find_dwg_to_include(self, item, migration_matrix):
        """Find which divergence with gene flow events we whould include for the demography, 'item'.
        Include any events for which the two populations derived from the same ancestor before
        time zero and there is migration allowed between the populations."""

        to_include = []

        # iterate over migration matrix and find populations with migration
        for index, row in migration_matrix.iterrows():
            for colname, value in row.items():
                include = False

                if index != colname and _allows_migration(value):

                    # check that the population diverged from its ancestor at a time > 0
                    for event in item.events:
                        if index in event.derived and colname in event.derived and event.time != 0:
                            include = True

                # if criteria are met, include the event.
                if include:
                    to_include.append((index,colname))

        return to_include

    def _draw_parameters_divergence(self, divergence_times, population_sizes, \
                                    divergence_demographies):
        """Draw parameters for divergence models."""

        models_with_parameters = []

        for original_model in divergence_demographies:

            this_model_with_parameters = []

            population_size_draws, population_size_keys = self._draw_population_sizes(
                original_model, population_sizes)
            divergence_time_draws = self._draw_divergence_times(
                population_size_draws, original_model, divergence_times)

            template = pickle.dumps(original_model)
            for rep in range(self.config['replicates']):
                model = pickle.loads(template)
                for population in model.populations:
                    population.initial_size = population_size_draws[population.name][rep]
                for event in model.events:
                    if hasattr(event, 'ancestral'):
                        event.time = divergence_time_draws[event.ancestral][rep]
                model.sort_events()
                this_model_with_parameters.append(model)

            models_with_parameters.append(this_model_with_parameters)

        return models_with_parameters

    def _draw_parameters_sc(self, divergence_times, population_sizes, sc_demographies):
        """Draw parameters for secondary contact models."""

        models_with_parameters = []

        for original_model in sc_demographies:

            this_model_with_parameters = []

            population_size_draws, population_size_keys = \
                self._draw_population_sizes(original_model, population_sizes)
            divergence_time_draws = self._draw_divergence_times(
                population_size_draws, original_model, divergence_times)
            migration_rate_draws = self._draw_migration_rates(
                population_size_keys, original_model, population_size_draws)
            migration_stop = self._get_migration_stops(divergence_time_draws)

            template = pickle.dumps(original_model)
            for rep in range(self.config['replicates']):
                model = pickle.loads(template)
                for population in model.populations:
                    population.initial_size = population_size_draws[population.name][rep]
                for event in model.events:
                    if hasattr(event, 'ancestral'):
                        event.time = divergence_time_draws[event.ancestral][rep]
                    elif hasattr(event, 'rate'):
                        event.time = migration_stop[rep]
                for key in migration_rate_draws:
                    model.migration_matrix[int(str(key.split('_')[0])),int(str(key.split('_')[1]))]\
                        = migration_rate_draws[key][rep]
                model.sort_events()
                this_model_with_parameters.append(model)

            models_with_parameters.append(this_model_with_parameters)
        return models_with_parameters

    def _draw_parameters_dwg(self, divergence_times, population_sizes, dwg_demographies):
        """Draw parameters for divergence with gene flow models."""

        models_with_parameters = []

        for original_model in dwg_demographies:

            this_model_with_parameters = []

            population_size_draws, population_size_keys = self._draw_population_sizes(
                original_model, population_sizes)
            divergence_time_draws = self._draw_divergence_times(
                population_size_draws, original_model, divergence_times)
            migration_rate_draws = self._draw_migration_rates(
                population_size_keys, original_model, population_size_draws)
            migration_start = self._get_migration_starts(
                original_model, divergence_time_draws, population_size_keys)

            template = pickle.dumps(original_model)
            for rep in range(self.config['replicates']):
                model = pickle.loads(template)
                for population in model.populations:
                    population.initial_size = population_size_draws[population.name][rep]
                for event in model.events:
                    if hasattr(event, 'ancestral'):
                        event.time = divergence_time_draws[event.ancestral][rep]
                    elif hasattr(event, 'rate'):
                        key = f"{population_size_keys[event.source]}_{population_size_keys[event.dest]}"
                        event.time = migration_start[key][rep]
                        event.rate = migration_rate_draws[key][rep]

                model.sort_events()
                this_model_with_parameters.append(model)

            models_with_parameters.append(this_model_with_parameters)

        return models_with_parameters

    def _draw_population_sizes(self, model, population_sizes):
        """Draw population sizes from priors."""

        population_size_draws = {}
        population_size_keys = {}

        if self.config['constant Ne']:
            min_size, max_size = population_sizes[list(population_sizes.keys())[0]]
            the_population_size = np.round(self.rng.uniform(
                low=min_size, high=max_size, size=self.config['replicates']),0)
            for index, population in enumerate(model.populations):
                population_size_draws[population.name] = the_population_size
                population_size_keys[population.name] = index
        else:

            for index, population in enumerate(model.populations):
                min_size, max_size = population_sizes[population.name]
                population_size_draws[population.name] = np.round(self.rng.uniform(
                    low=min_size, high=max_size, size=self.config['replicates']),0)
                population_size_keys[population.name] = index

        return(population_size_draws, population_size_keys)

    def _draw_divergence_times(self, population_size_draws, model, divergence_times):
        """Draw divergence times from priors."""

        divergence_time_draws = {}

        # create a list of populations that are never ancestors
        all_populations = population_size_draws.keys()
        ancestral = [x for x in model.events if hasattr(x, 'ancestral')]
        ancestral = [x.ancestral for x in ancestral]
        non_ancestral = list(set(all_populations)- set(ancestral))

        # add zero for non-ancestral populations
        for x in non_ancestral:
            divergence_time_draws[x] = np.repeat(0, self.config['replicates'])

        for event in model.events:
            if hasattr(event, 'ancestral'):
                if event.time != 0:
                    # get the divergence times of the derived populations,
                    # and use that to set the minimum value for drawing divergence times
                    min_values = np.maximum.reduce([
                        divergence_time_draws[event.derived[0]],
                        divergence_time_draws[event.derived[1]],
                        np.repeat(divergence_times[event.ancestral][0], self.config['replicates'])])
                    divergence_time_draws[event.ancestral] = np.round(self.rng.uniform(
                        low=min_values, high=divergence_times[event.ancestral][1]), 0)
                elif event.time == 0:
                    divergence_time_draws[event.ancestral] = np.repeat(0, self.config['replicates'])

        return divergence_time_draws

    def _draw_migration_rates(self, population_size_keys, model, population_size_draws):
        """Draw migration rates from priors."""

        prior = self.config["migration rate"]
        use_nm = not all(isinstance(x, float) for x in prior)
        if use_nm:
            if not ("Nm" in prior[0] and "Nm" in prior[1]):
                raise ValueError("Error: Inconsistent use of Nm in migration rate prior.")
            low_value, high_value = float(prior[0].strip('Nm')), float(prior[1].strip('Nm'))
        else:
            low_value, high_value = prior[0], prior[1]

        draws = {}
        directions = []
        for event in model.events:
            if hasattr(event, 'rate'):
                pair = frozenset((event.source, event.dest)) if self.config['symmetric'] \
                    else (event.source, event.dest)
                draws[pair] = np.round(self.rng.uniform(
                    low=low_value, high=high_value, size=self.config["replicates"]), 10)
                directions.append((event.source, event.dest, pair))

        migration_rate_draws = {}
        for source, dest, pair in directions:
            key = f"{population_size_keys[source]}_{population_size_keys[dest]}"
            if use_nm:
                migration_rate_draws[key] = draws[pair] / population_size_draws[source]
            else:
                migration_rate_draws[key] = draws[pair]
        return migration_rate_draws

    def _get_migration_stops(self, divergence_time_draws):
        """Get stop times for migration in the secondary contact models."""

        minimum_divergence = []
        for rep in range(self.config["replicates"]):
            min_div = np.inf
            for key in divergence_time_draws:
                if divergence_time_draws[key][rep] > 0 and \
                    divergence_time_draws[key][rep] < min_div:
                    min_div = divergence_time_draws[key][rep]
            minimum_divergence.append(min_div)
        migration_stop = [np.ceil(x/2) for x in minimum_divergence]
        return migration_stop

    def _get_migration_starts(self, model, divergence_time_draws, population_size_keys):
        """Get start times for migration in the divergence with gene flow models."""

        migration_start = {}

        for rep in range(self.config['replicates']):

            for event in model.events:

                if hasattr(event, 'rate'):
                    daughter1, daughter2 = event.source, event.dest

                    for divevent in model.events:
                        if hasattr(divevent, 'ancestral'):
                            if daughter1 in divevent.derived and daughter2 in divevent.derived:
                                ancestor = divevent.ancestral

                    tdiv_ancestor = divergence_time_draws[ancestor][rep]
                    tdiv_daughter1 = divergence_time_draws[daughter1][rep]
                    tdiv_daughter2 = divergence_time_draws[daughter2][rep]
                    startime = ((tdiv_ancestor-max(tdiv_daughter1, tdiv_daughter2))/2) \
                        + max(tdiv_daughter1, tdiv_daughter2)

                    try:
                        migration_start[f"{population_size_keys[daughter1]}_{population_size_keys[daughter2]}"].append(startime)
                    except Exception:
                        migration_start[f"{population_size_keys[daughter1]}_{population_size_keys[daughter2]}"] = []
                        migration_start[f"{population_size_keys[daughter1]}_{population_size_keys[daughter2]}"].append(startime)

        return migration_start

    def _plot_models(self, demographies, labels, outplot):
        """Plot example models for a given type of demography."""

        if outplot is None:
            for modelix, model in enumerate(demographies):
                if modelix % self.config['replicates'] == 0:
                    new_model = copy.deepcopy(model)
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
                        graph = new_model.to_demes()

                        # Plot the model
                        fig, ax = plt.subplots()
                        demesdraw.tubes(graph, ax=ax, seed=1)
                        plt.title(f"Model: {labels[modelix]}")
                        pdf.savefig(fig)
                        plt.close(fig)

def _get_priors(tree):
    """Get priors for population sizes and divergence times from the species tree."""
    # build dictionary of priors for population sizes
    population_sizes = {}
    divergence_times = {}

    # get priors from species tree
    try:
        for node in tree.postorder_node_iter():
            min_ne, max_ne = map(int, node.annotations['ne'].value.strip("'").split("-"))
            if node.is_leaf():
                population_sizes[node.taxon.label.strip("'")] = [min_ne, max_ne]
            else:
                node_label = node.label.strip("'")
                population_sizes[node_label] = [min_ne, max_ne]
                min_div, max_div = map(int, node.annotations['div'].value.strip("'").split("-"))
                divergence_times[node_label] = [min_div, max_div]

    except (ValueError) as ve:
        raise ValueError(f"Error: Issue when getting priors from species tree: {ve}") from ve
    except (KeyError) as ke:
        raise KeyError(f"Error: Issue when getting priors from species tree: {ke}") from ke
    except Exception as e:
        raise RuntimeError("Unexpected Error: Issue when getting priors from species tree: "
                           f"{e}") from e

    _check_priors(tree, population_sizes, divergence_times)


    return(population_sizes, divergence_times)

def _check_priors(tree, population_sizes, divergence_times):
    """Raise a clear error for priors that cannot produce a valid model."""

    for name, (low, high) in population_sizes.items():
        if not 0 < low <= high:
            raise ValueError(f"Error: Ne prior for {name} ({low}-{high}) must be positive, "
                             "with the minimum no larger than the maximum.")

    for name, (low, high) in divergence_times.items():
        if not 0 <= low <= high:
            raise ValueError(f"Error: divergence time prior for {name} ({low}-{high}) must be "
                             "non-negative, with the minimum no larger than the maximum.")

    # an ancestor is drawn between its oldest child's time and its own maximum,
    # so no child may have a maximum older than its ancestor's maximum
    for node in tree.postorder_internal_node_iter():
        parent = node.parent_node
        if parent is None:
            continue
        child_name = node.label.strip("'")
        parent_name = parent.label.strip("'")
        child_max = divergence_times[child_name][1]
        parent_max = divergence_times[parent_name][1]
        if child_max > parent_max:
            raise ValueError(
                f"Error: the divergence time prior for {child_name} (maximum {child_max}) "
                f"is older than the maximum for its ancestor {parent_name} ({parent_max}), "
                f"so {parent_name} could be drawn younger than {child_name}. Raise the "
                f"maximum for {parent_name} or lower the maximum for {child_name}.")

def _divergence_model_key(demography):
    """Structure of a divergence model, independent of population and event order."""
    populations = frozenset(p.name for p in demography.populations)
    splits = frozenset((e.ancestral, frozenset(e.derived))
                       for e in demography.events if hasattr(e, 'ancestral'))
    return populations, splits

def _allows_migration(value):
    """Read one migration matrix entry. pandas keeps T/F as text but turns a matrix
    written entirely in TRUE/FALSE into booleans, so accept both."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in ("T", "TRUE"):
        return True
    if text in ("F", "FALSE"):
        return False
    raise ValueError(f"Error: migration matrix entries must be T or F, not {value!r}.")