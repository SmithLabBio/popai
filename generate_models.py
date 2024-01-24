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

            # output directory
            if line.startswith("output directory ="):
                output_directory = line.split("=")[1].split("#")[0].strip()


    return(species_tree, replicates, migration_df, symmetric, maxmig, secondary, dwg, migration_rate, output_directory)

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
        # add populations and set intial size to 1000 (this will be edited in a later stage)
        demography = add_populations(demography, species_tree)
        # add divergences and set initial size to 1000 if the node is not collapsed, and 0 otherwise
        for internal_node in species_tree.postorder_internal_node_iter():
            derived_1 = internal_node.child_nodes()[0].label
            derived_2 = internal_node.child_nodes()[1].label
            if derived_1 == None:
                derived_1 = str(internal_node.child_nodes()[0].taxon).strip("'")
            if derived_2 == None:
                derived_2 = str(internal_node.child_nodes()[1].taxon).strip("'")
            ancestral = str(internal_node.label).strip("'")
            if internal_node in combo:
                demography.add_population_split(time=0, derived=[derived_1, derived_2], ancestral=ancestral)
            else:
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

def model_to_yaml_divergence(model, outputfile):

    yaml_dict = {}

    for population in model.populations:
        yaml_dict[population.name] = {'size':population.initial_size}

    for event in model.events:
        yaml_dict[event.ancestral]['end_time']= event.time
        yaml_dict[event.derived[0]]['ancestor'] = event.ancestral
        yaml_dict[event.derived[1]]['ancestor'] = event.ancestral
    

    with open(outputfile, 'w') as f:
        f.write('time_units: generations\n')
        f.write('demes:\n')
        for item in yaml_dict:
            f.write('  -name: %s\n' % item)
            try:
                f.write('    ancestors:[%s]\n' % yaml_dict[item]["ancestor"])
            except:
                pass
            f.write('    epochs:\n')
            try:
                f.write('      -{end_time: %s, start_size: %s}\n' % (yaml_dict[item]["end_time"], yaml_dict[item]["size"]))
            except:
                f.write('      -{end_time: 0, start_size: %s}\n' % (yaml_dict[item]["size"]))

def model_to_yaml_sc(model, outputfile, migration_matrix, migration_prior):

    yaml_dict = {}

    for population in model.populations:
        yaml_dict[population.name] = {'size':population.initial_size}

    for event in model.events:
        if event.time != 0 and hasattr(event, 'ancestral'):
            yaml_dict[event.ancestral]['end_time']= event.time
            yaml_dict[event.derived[0]]['ancestor'] = event.ancestral
            yaml_dict[event.derived[1]]['ancestor'] = event.ancestral
    

    with open(outputfile, 'w') as f:
        f.write('time_units: generations\n')
        f.write('demes:\n')
        for item in yaml_dict:
            f.write('  -name: %s\n' % item)
            try:
                f.write('    ancestors:[%s]\n' % yaml_dict[item]["ancestor"])
            except:
                pass
            f.write('    epochs:\n')
            try:
                f.write('      -{end_time: %s, start_size: %s}\n' % (yaml_dict[item]["end_time"], yaml_dict[item]["size"]))
            except:
                f.write('      -{end_time: 0, start_size: %s}\n' % (yaml_dict[item]["size"]))

        f.write('migrations:\n')
        row_count=0
        for row in migration_matrix:
            col_count=0
            for col in row:
                if col != 0:
                    f.write('{demes: [%s, %s], rate: %s, start_time: 0, end_time: mindiv/2}\n' % (row_count, col_count, migration_prior))
                col_count+=1
            row_count+=1

def model_to_yaml_dwg(model, outputfile, migration_matrix, migration_prior):

    yaml_dict = {}

    for population in model.populations:
        yaml_dict[population.name] = {'size':population.initial_size}

    for event in model.events:
        if event.time != 0 and hasattr(event, 'ancestral'):
            yaml_dict[event.ancestral]['end_time']= event.time
            yaml_dict[event.derived[0]]['ancestor'] = event.ancestral
            yaml_dict[event.derived[1]]['ancestor'] = event.ancestral

    with open(outputfile, 'w') as f:
        f.write('time_units: generations\n')
        f.write('demes:\n')
        for item in yaml_dict:
            f.write('  -name: %s\n' % item)
            try:
                f.write('    ancestors:[%s]\n' % yaml_dict[item]["ancestor"])
            except:
                pass
            f.write('    epochs:\n')
            try:
                f.write('      -{end_time: %s, start_size: %s}\n' % (yaml_dict[item]["end_time"], yaml_dict[item]["size"]))
            except:
                f.write('      -{end_time: 0, start_size: %s}\n' % (yaml_dict[item]["size"]))

    for event in model.events:
        if event.time != 0 and hasattr(event, 'rate'):
            deme_1 = event.populations[0]
            deme_2 = event.populations[1]
            # find the ancestor of the two focal populations to figure out when to start and end migration
            deme_1_ancestor = yaml_dict[deme_1]['ancestor']
            deme_2_ancestor = yaml_dict[deme_2]['ancestor']
            if not deme_1_ancestor == deme_2_ancestor:
                sys.exit("ERROR IN DWG MODEL.")


#        row_count=0
#        for row in migration_matrix:
#            col_count=0
#            for col in row:
#                if col != 0:
#                    print(row,col)
#                    f.write('{demes: [%s, %s], rate: %s, start_time: 0, end_time: mindiv/2}\n' % (row_count, col_count, migration_prior))
#                col_count+=1
#            row_count+=1


def create_yaml_files(output_directory, baseline_demographies, sc_demographies, dwg_demographies, population_sizes, divergence_times, migration_rates):

    print("Creating yaml model files.")
    modno = 1

    # for divergence only models
    for model in baseline_demographies:
        
        # put priors in for population sizes
        for population in model.populations:
            prior = 'U(%s, %s)' % (population_sizes[population.name][0], population_sizes[population.name][1])
            population.initial_size=prior
        
        # put priors in for divergence times
        for event in model.events:
            if event.time != 0:
                prior = 'U(%s, %s)' % (divergence_times[event.ancestral][0],divergence_times[event.ancestral][1])
                event.time = prior

        # write to yaml
        model_to_yaml_divergence(model, '%s/model_%s.yaml' % (output_directory, str(modno)))

        modno += 1

    for model in sc_demographies:

        # put priors in for population sizes
        for population in model.populations:
            prior = 'U(%s, %s)' % (population_sizes[population.name][0], population_sizes[population.name][1])
            population.initial_size=prior
        
        # put priors in for divergence times
        for event in model.events:
            if event.time != 0 and hasattr(event, 'ancestral'):
                prior = 'U(%s, %s)' % (divergence_times[event.ancestral][0],divergence_times[event.ancestral][1])
                event.time = prior

        # write to yaml
        model_to_yaml_sc(model, '%s/model_%s.yaml' % (output_directory, str(modno)), migration_matrix=model.migration_matrix, migration_prior=migration_rates)

        modno += 1

    for model in dwg_demographies:

        # put priors in for population sizes
        for population in model.populations:
            prior = 'U(%s, %s)' % (population_sizes[population.name][0], population_sizes[population.name][1])
            population.initial_size=prior
        
        # put priors in for divergence times
        for event in model.events:
            if event.time != 0 and hasattr(event, 'ancestral'):
                prior = 'U(%s, %s)' % (divergence_times[event.ancestral][0],divergence_times[event.ancestral][1])
                event.time = prior

        # write to yaml
        model_to_yaml_dwg(model, '%s/model_%s.yaml' % (output_directory, str(modno)), migration_matrix=model.migration_matrix, migration_prior=migration_rates)

        modno += 1


def main():

    # parse the user arguments
    options, args = parse_arguments()

    # get information from the configuration file
    species_tree, replicates, migration_df, symmetric, maxmig, secondary, dwg, migration_rate, output_directory = parse_config(configfile=options.configfile)

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

    # create modified yaml files
    create_yaml_files(output_directory, baseline_demographies, sc_demographies, dwg_demographies, population_sizes, divergence_times, migration_rate)


    # to do:
    # 1. add random seeds
    # 2. take yaml formatted files as input
    # 3. write yaml formatted files as output (I would do this prior to simulations, and draw parameters from these files.)
    # 4. COMPLETE add secondary contact histories
    # 5. COMPLETE add divergence with gene flow histories
    # 6. perform simulations
    # 7. option to save simulated data in useful format
    # 8. create input for CNN
    # 9. create input for RF with SFS
    #10. crete input for RF with sumstats
    #11. create input for CNN with sumstats
    #12. create input for CNN with SFS
    #13. create function to train CNN
    #14. create function to train RF with SFS
    #15. create function to train RF with sumstats
    #16. create function to train CNN with SFS
    #17. create function to train CNN with sumstats
    #18. create function to output training information.
    #18. create function to apply trained models.
    #19. create functioin to output results.
    #20. test on simulated datasets for a variety of model setups.
    #21. write preprint
    #22. create documentation
    #23. turn into python package
    #24. add migration rate priors
    #25. parameter estimation?
    #26. make sure parameter draws play by the rules of the tree
    #27. does symmetric mean that we use the same rate, or just that migration happens in both directions?
    #28. secondary contact only occurs between current populations (including ancestral populations in the tree when nodes are collapsed), and dwg only occurs between populations that share a most recent common ancestor



if __name__ == "__main__":
    main()
