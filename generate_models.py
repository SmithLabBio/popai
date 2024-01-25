from optparse import OptionParser
#import ete3
import dendropy
import msprime
import copy
from itertools import chain, combinations
import numpy as np
import pandas as pd
import os
import demes
import sys
import random # only used for demes plotting
import demesdraw
import matplotlib.pyplot as plt


def parse_arguments():
    # get command line input from user using optparse.
    parser = OptionParser()
    
    parser.add_option("-c","--configfile", help="Path to configuration file with parameters.",
                        action="store", type="string", dest="configfile")

    (options, args) = parser.parse_args()
    
    if not options.configfile:
        parser.error("Config file path is required. Use -c or --configfile option to specify the path.")
    return(options, args)

def parse_config(configfile):

    # open the config file
    with open(configfile, 'r') as f:

        for line in f.readlines():

            # read the nexus formatted species tree with dendropy.
            if line.startswith("species tree ="):
                species_tree = dendropy.Tree.get(path=line.split("=")[1].split("#")[0].strip(), schema="nexus")

            # get the number of replicates
            if line.startswith("replicates ="):
                replicates = int(line.split("=")[1].split("#")[0].strip())

            # get the migration matrix
            if line.startswith("migration matrix ="):
                if not line.split("=")[1].strip() == "None":
                    migration_df = pd.read_csv(line.split("=")[1].split("#")[0].strip(), index_col=0)

            # is gene flow constrained to be symmetric?
            if line.startswith("symmetric ="):
                symmetric = line.split("=")[1].split("#")[0].strip()

            # max migration events
            if line.startswith("max migration events ="):
                maxmig = int(line.split("=")[1].split("#")[0].strip())

            # secondary contact
            if line.startswith("secondary contact ="):
                secondary = line.split("=")[1].split("#")[0].strip()

            # divergence with gene flow
            if line.startswith("divergence with gene flow ="):
                dwg = line.split("=")[1].split("#")[0].strip()

            # migration rate
            if line.startswith("migration rate ="):
                migration_rate = line.split("=")[1].split("#")[0].strip()
                migration_rate = [float(migration_rate.split("U(")[1].split(",")[0]), float(migration_rate.split(",")[1].split(")")[0].strip())]

            # output directory
            if line.startswith("output directory ="):
                output_directory = line.split("=")[1].split("#")[0].strip()

            # random seed
            if line.startswith("seed ="):
                seed = int(line.split("=")[1].split("#")[0].strip())



    return(species_tree, replicates, migration_df, symmetric, maxmig, secondary, dwg, migration_rate, output_directory, seed)

def get_priors(species_tree):
    """A function to get priors for population sizes and divergence times from the species tree."""

    # build dictionary of priors for population sizes
    population_sizes = {}
    for leaf in species_tree.leaf_node_iter():
        min_ne = int(str(leaf.annotations['ne']).split("=")[1].strip("'").split("-")[0])
        max_ne = int(str(leaf.annotations['ne']).split("=")[1].strip("'").split("-")[1])
        population_sizes[str(leaf.taxon).strip("'")] = [min_ne, max_ne]
    for internal_node in species_tree.postorder_internal_node_iter():
        min_ne = int(str(internal_node.annotations['ne']).split("=")[1].strip("'").split("-")[0])
        max_ne = int(str(internal_node.annotations['ne']).split("=")[1].strip("'").split("-")[1])
        population_sizes[str(internal_node.label).strip("'")] = [min_ne, max_ne]

    # build dictionary of priors for divergence times
    divergence_times = {}
    for internal_node in species_tree.postorder_internal_node_iter():
        min_div = int(str(internal_node.annotations['div']).split("=")[1].strip("'").split("-")[0])
        max_div = int(str(internal_node.annotations['div']).split("=")[1].strip("'").split("-")[1])
        divergence_times[str(internal_node.label).strip("'")] = [min_div, max_div]

    return(population_sizes, divergence_times)

def add_populations(demography, species_tree):
    # add populations and set intial size to 1000 (this will be edited in a later stage)
    for leaf in species_tree.leaf_node_iter():
        demography.add_population(name=str(leaf.taxon).strip("'"), initial_size=1000)
    for internal_node in species_tree.postorder_internal_node_iter():
        demography.add_population(name=str(internal_node.label).strip("'"), initial_size=1000)
    return(demography)

def remove_conflicting(all_combos):
    """Remove any combos of nodes to collapse that are conflicting, meaning that daughter nodes of collapsed nodes are not collapsed."""
    keep_combos = []
    for item in all_combos:
        keep = True
        for subitem in item:
            for child in subitem.postorder_internal_node_iter():
                if child not in item:
                    keep = False
        if keep:
            keep_combos.append(item)
    return(keep_combos)

def get_derived_populations(internal_node):
    derived_1 = internal_node.child_nodes()[0].label
    derived_2 = internal_node.child_nodes()[1].label
    if derived_1 == None:
        derived_1 = str(internal_node.child_nodes()[0].taxon).strip("'")
    if derived_2 == None:
        derived_2 = str(internal_node.child_nodes()[1].taxon).strip("'")
    return(derived_1, derived_2)

def create_baseline_demographies(species_tree):

    """Here we create our baseline models with the correct population divergences set to zero, and migration where needed. We will draw other parameters from priors later."""

    demographies = []

    # get a list of collapsable nodes, and then get all potential combos of these
    collapsable_nodes = species_tree.internal_nodes()
    all_combos = chain.from_iterable(combinations(collapsable_nodes, r) for r in range(1, len(collapsable_nodes) + 1))
    all_combos = [list(combo) for combo in all_combos]

    # remove conflicting combos, to get a list of the models to include (minus the full model)
    all_combos = remove_conflicting(all_combos)
    all_combos.append([])

    # now generate the demographies for all combos
    for combo in all_combos:
        demography = msprime.Demography()
        added_populations = []

        # add populations that should be added, and set initial sizes (these will be changed later.)
        for internal_node in species_tree.postorder_internal_node_iter():
            derived_1, derived_2 = get_derived_populations(internal_node)
            ancestral = str(internal_node.label).strip("'")
            if internal_node not in combo:
                demography.add_population(name=derived_1, initial_size=1000)
                demography.add_population(name=derived_2, initial_size=1000)
        demography.add_population(name=str(species_tree.seed_node.label).strip("'"), initial_size=1000)

        # add non-zero divergence events
        for internal_node in species_tree.postorder_internal_node_iter():
            if internal_node not in combo:
                derived_1, derived_2 = get_derived_populations(internal_node)
                ancestral = str(internal_node.label).strip("'")
                demography.add_population_split(time=1000, derived=[derived_1, derived_2], ancestral=ancestral)

        demographies.append(demography)
        del(demography)

    return(demographies)

def find_sc_to_include(item, migration_df):
    """Find which secondary contact events we whould include for the demography, 'item'"""
    to_include = []
    
    # iterate over migration matrix and find populations with migration
    for index, row in migration_df.iterrows():
        for colname in migration_df.columns:
            include = [False,False]
            if index != colname and row[colname]=="T":

                # check that the population diverged from its ancestor at a time > 0
                considering = [index,colname]
                for population in range(len(considering)):
                    for event in item.events:
                        if considering[population] in event.derived:
                            if event.time==0:
                                include[population] = False
                            else:
                                include[population] = True
                    # check that the ppulation exists in the present.
                    for event in item.events:
                        if considering[population] == event.ancestral:
                            if event.time!=0:
                                include[population] = False
            # if both poulations meet are criteria, include the event.
            if include[0] and include[1]:
                to_include.append((index,colname))
    
    return(to_include)

def find_dwg_to_include(item, migration_df):
    """Find which divergence with gene flow events we whould include for the demography, 'item'"""
    to_include = []
    
    # iterate over migration matrix and find populations with migration
    for index, row in migration_df.iterrows():
        for colname in migration_df.columns:
            include = False
            if index != colname and row[colname]=="T":

                # check that the population diverged from its ancestor at a time > 0
                for event in item.events:
                    if index in event.derived and colname in event.derived:
                        if event.time==0:
                            include = False
                        else:
                                include = True

            # if both poulations meet are criteria, include the event.
            if include:
                to_include.append((index,colname))
    
    return(to_include)

def add_sc_demographies(baseline_demographies, migration_df, symmetric, maxmig):

    migration_demographies = []

    # iterate over demographies
    for item in baseline_demographies:

        # list of events to include
        to_include = set(find_sc_to_include(item, migration_df))

        # keep only one per pair if symmetric rates are enforced
        if symmetric == 'True':
            to_include = {frozenset(pair) for pair in to_include}
        
        to_include = list(to_include)

        # get all combos of events
        combos_of_migration = chain.from_iterable(combinations(to_include, r) for r in range(1, min(maxmig,len(to_include)) + 1))
        combos_of_migration = [list(combo) for combo in combos_of_migration]

        # create histories with each combo of events
        for combo in combos_of_migration:
            migration_demography = copy.deepcopy(item)
            
            # if symmetric true, then add symmetric migration and add ceasing of migration
            if symmetric == "True":
                for populationpair in combo:
                    migration_demography.set_symmetric_migration_rate(populationpair, 1e-3)
                    migration_demography.add_symmetric_migration_rate_change(100, list(populationpair), 0)
            
            # if not symmetric then do asymmetric
            else:
                for populationpair in combo:
                    populationpairlist = list(populationpair)
                    migration_demography.set_migration_rate(source=populationpairlist[0], dest=populationpairlist[1], rate=1e-3)
                    migration_demography.add_migration_rate_change(time=100, source=populationpairlist[0], dest=populationpairlist[1], rate=0)

            # sort the events and add the demography to the list
            migration_demography.sort_events()
            migration_demographies.append(migration_demography)
        
    return(migration_demographies)

def add_dwg_demographies(baseline_demographies, migration_df, symmetric, maxmig):

    migration_demographies = []

    # iterate over demographies
    for item in baseline_demographies:

        # list of events to include
        to_include = set(find_dwg_to_include(item, migration_df))

        # keep only one per pair if symmetric rates are enforced
        if symmetric == "True":
            to_include = {frozenset(pair) for pair in to_include}
        
        to_include = list(to_include)

        # get all combos of events
        combos_of_migration = chain.from_iterable(combinations(to_include, r) for r in range(1, min(maxmig,len(to_include)) + 1))
        combos_of_migration = [list(combo) for combo in combos_of_migration]
        
        # create histories with each combo of events
        for combo in combos_of_migration:
            migration_demography = copy.deepcopy(item)
            
            # if symmetric true, then add symmetric migration and add ceasing of migration
            if symmetric == "True":
                for populationpair in combo:
                    migration_demography.add_symmetric_migration_rate_change(100, list(populationpair), rate=1e-3)
            
            # if not symmetric then do asymmetric
            else:
                for populationpair in combo:
                    populationpairlist = list(populationpair)
                    migration_demography.add_migration_rate_change(time=100, source=populationpairlist[0], dest=populationpairlist[1], rate=1e-3)

            # sort the events and add the demography to the list
            migration_demography.sort_events()
            migration_demographies.append(migration_demography)

        
    return(migration_demographies)

def draw_population_sizes(model, population_sizes, replicates, rng):
    # draw population sizes and map populations to keys
    population_size_draws = {}
    population_size_keys = {}
    count=0
    for population in model.populations:
        population_size_draws[population.name] = rng.uniform(low=population_sizes[population.name][0], high=population_sizes[population.name][1], size=replicates)
        population_size_keys[population.name] = count
        count+=1
    return(population_size_draws, population_size_keys)

def draw_divergence_times(population_size_draws, model, divergence_times, replicates, rng):

    # create a list of populations that are never ancestors
    all_populations = population_size_draws.keys()
    ancestral = [x for x in model.events if hasattr(x, 'ancestral')]
    ancestral = [x.ancestral for x in ancestral]
    non_ancestral = list(set(all_populations)- set(ancestral))
    # draw divergence times
    divergence_time_draws = {}
    # add zero for non-ancestral populations
    for x in non_ancestral:
        divergence_time_draws[x] = np.repeat(0, replicates)
    for event in model.events:
        if hasattr(event, 'ancestral'):
            if event.time != 0:
                # get the divergence times of the derived populations, and use that to set the minimum value for drawing divergence times
                min_values = [max(x) for x in zip(divergence_time_draws[event.derived[0]], divergence_time_draws[event.derived[1]], np.repeat(divergence_times[event.ancestral][0], replicates))]
                divergence_time_draws[event.ancestral] = [rng.uniform(low=x, high=divergence_times[event.ancestral][1]) for x in min_values]
            elif event.time == 0:
                divergence_time_draws[event.ancestral] = np.repeat(0, replicates)
    return(divergence_time_draws)

def draw_migration_rates(population_size_keys, model, migration_rate, replicates, symmetric, rng):

    # draw migration rates
    migration_rate_draws = {}
    for event in model.events:
        if hasattr(event, 'rate'):
            if symmetric=='True':
                migration_rate_draws["%s_%s" % (population_size_keys[event.populations[0]], population_size_keys[event.populations[1]])] =  rng.uniform(low=migration_rate[0], high=migration_rate[1], size=replicates)
            else:
                migration_rate_draws["%s_%s" % (population_size_keys[event.source], population_size_keys[event.dest])] =  rng.uniform(low=migration_rate[0], high=migration_rate[1], size=replicates)
    return(migration_rate_draws)

def get_migration_stops(replicates, divergence_time_draws):
    minimum_divergence = []
    for rep in range(replicates):
        min = np.inf
        for key in divergence_time_draws:
            if divergence_time_draws[key][rep] > 0 and divergence_time_draws[key][rep] < min:
                min = divergence_time_draws[key][rep]
        minimum_divergence.append(min)
    migration_stop = [np.ceil(x/2) for x in minimum_divergence]
    return(migration_stop)

def get_migration_starts(model, replicates, divergence_time_draws, population_size_keys, symmetric):
    migration_start = {}
    for rep in range(replicates):
        for event in model.events:
            if hasattr(event, 'rate'):
                if symmetric == "True":
                    daughter1 = event.populations[0]
                    daughter2 = event.populations[1]
                else:
                    daughter1 = event.source
                    daughter2 = event.dest

                for divevent in model.events:
                    if hasattr(divevent, 'ancestral'):
                        if daughter1 in divevent.derived and daughter2 in divevent.derived:
                            ancestor = divevent.ancestral
                tdiv_ancestor = divergence_time_draws[ancestor][rep]
                tdiv_daughter1 = divergence_time_draws[daughter1][rep]
                tdiv_daughter2 = divergence_time_draws[daughter2][rep]
                startime = ((tdiv_ancestor-max(tdiv_daughter1, tdiv_daughter2))/2) + max(tdiv_daughter1, tdiv_daughter2)
                try:
                    migration_start["%s_%s" % (population_size_keys[daughter1], population_size_keys[daughter2])].append(startime)
                except:
                    migration_start["%s_%s" % (population_size_keys[daughter1], population_size_keys[daughter2])] = []
                    migration_start["%s_%s" % (population_size_keys[daughter1], population_size_keys[daughter2])].append(startime)
    return(migration_start)

def draw_parameters_baseline(demographies, divergence_times, population_sizes, replicates, rng):
    models_with_parameters = []

    """Draw parameters for models."""
    for original_model in demographies:

        this_model_with_parameters = []

        population_size_draws, population_size_keys = draw_population_sizes(original_model, population_sizes, replicates, rng)
        
        divergence_time_draws = draw_divergence_times(population_size_draws, original_model, divergence_times, replicates, rng)

        for rep in range(replicates):
            model = copy.deepcopy(original_model)
            for population in model.populations:
                population.initial_size = population_size_draws[population.name][rep]
            for event in model.events:
                if hasattr(event, 'ancestral'):
                    event.time = divergence_time_draws[event.ancestral][rep]

            this_model_with_parameters.append(model)
            del(model)

        models_with_parameters.append(this_model_with_parameters)

    return(models_with_parameters)

def draw_parameters_sc(demographies, divergence_times, population_sizes, replicates, migration_rate, symmetric, rng):
    models_with_parameters = []

    """Draw parameters for models."""
    for original_model in demographies:

        this_model_with_parameters = []

        population_size_draws, population_size_keys = draw_population_sizes(original_model, population_sizes, replicates, rng)

        divergence_time_draws = draw_divergence_times(population_size_draws, original_model,divergence_times, replicates, rng)

        migration_rate_draws = draw_migration_rates(population_size_keys, original_model, migration_rate, replicates, symmetric, rng)

        migration_stop = get_migration_stops(replicates, divergence_time_draws)

        for rep in range(replicates):
            model = copy.deepcopy(original_model)
            for population in model.populations:
                population.initial_size = population_size_draws[population.name][rep]
            for event in model.events:
                if hasattr(event, 'ancestral'):
                    event.time = divergence_time_draws[event.ancestral][rep]
                elif hasattr(event, 'rate'):
                    event.time = migration_stop[rep]
            for key in migration_rate_draws.keys():
                model.migration_matrix[int(str(key.split('_')[0])),int(str(key.split('_')[1]))] = migration_rate_draws[key][rep]
                if symmetric == "True":
                    model.migration_matrix[int(str(key.split('_')[1])),int(str(key.split('_')[0]))] = migration_rate_draws[key][rep]
            this_model_with_parameters.append(model)
            del(model)

        models_with_parameters.append(this_model_with_parameters)

    return(models_with_parameters)

def draw_parameters_dwg(demographies, divergence_times, population_sizes, replicates, migration_rate, symmetric, rng):
    models_with_parameters = []

    """Draw parameters for models."""
    for original_model in demographies:

        this_model_with_parameters = []

        population_size_draws, population_size_keys = draw_population_sizes(original_model, population_sizes, replicates, rng)

        divergence_time_draws = draw_divergence_times(population_size_draws, original_model,divergence_times, replicates, rng)

        migration_rate_draws = draw_migration_rates(population_size_keys, original_model, migration_rate, replicates, symmetric, rng)

        migration_start = get_migration_starts(original_model, replicates, divergence_time_draws, population_size_keys, symmetric)

        for rep in range(replicates):
            model = copy.deepcopy(original_model)
            for population in model.populations:
                population.initial_size = population_size_draws[population.name][rep]
            for event in model.events:
                if hasattr(event, 'ancestral'):
                    event.time = divergence_time_draws[event.ancestral][rep]
                elif hasattr(event, 'rate'):
                    if symmetric=="True":
                        event.time = migration_start["%s_%s" % (population_size_keys[event.populations[0]], population_size_keys[event.populations[1]])][rep]
                        event.rate = migration_rate_draws["%s_%s" % (population_size_keys[event.populations[0]], population_size_keys[event.populations[1]])][rep]
                    else:
                        event.time = migration_start["%s_%s" % (population_size_keys[event.source], population_size_keys[event.dest])][rep]
                        event.rate = migration_rate_draws["%s_%s" % (population_size_keys[event.source], population_size_keys[event.dest])][rep]

            model.sort_events()
            this_model_with_parameters.append(model)
            del(model)

        models_with_parameters.append(this_model_with_parameters)

    return(models_with_parameters)

def verify(demographies):
    for model in demographies:
        demo_to_plot = random.sample(model, 1)[0]
        graph = demo_to_plot.to_demes()
        fig, ax = plt.subplots()
        demesdraw.tubes(graph, ax=ax, seed=1)
        plt.show()

def main():

    # parse the user arguments
    options, args = parse_arguments()

    # get information from the configuration file
    species_tree, replicates, migration_df, symmetric, maxmig, secondary, dwg, migration_rate, output_directory, seed = parse_config(configfile=options.configfile)

    # create output directory
    os.system('mkdir -p %s' % output_directory)

    # get priors from species tree
    population_sizes, divergence_times = get_priors(species_tree=species_tree)

    # build a list of baseline msprime demographies (we will sample population sizes later)
    baseline_demographies = create_baseline_demographies(species_tree)

    # add migration histories for secondary contact scenarios
    if secondary == 'True':
        sc_demographies = add_sc_demographies(baseline_demographies, migration_df, symmetric, maxmig)
    elif secondary == 'False':
        sc_demographies = []

    # add migration histories for divergence with gene flow scenarios
    if dwg == 'True':
        dwg_demographies = add_dwg_demographies(baseline_demographies, migration_df, symmetric, maxmig)
    elif dwg == 'False':
        dwg_demographies = []

    print('Creating %s different models based on user input.' % len(baseline_demographies+sc_demographies+dwg_demographies))


    # draw parameters from priors for each set of demographies
    rng = np.random.default_rng(seed)

    # draw parameters for models
    parameterized_baseline_demographies = draw_parameters_baseline(demographies=baseline_demographies, divergence_times=divergence_times, population_sizes=population_sizes, replicates=replicates, rng=rng)
    parameterized_sc_demographies = draw_parameters_sc(demographies=sc_demographies, divergence_times=divergence_times, population_sizes=population_sizes, replicates=replicates, migration_rate=migration_rate, symmetric=symmetric, rng=rng)
    parameterized_dwg_demographies = draw_parameters_dwg(demographies=dwg_demographies, divergence_times=divergence_times, population_sizes=population_sizes, replicates=replicates, migration_rate=migration_rate, symmetric=symmetric, rng=rng)

    # write parameterized models to demes file
    verify(parameterized_baseline_demographies)
    verify(parameterized_sc_demographies)
    verify(parameterized_dwg_demographies)

if __name__ == "__main__":
    main()
