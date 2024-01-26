##############################
Building models with delimitpy
##############################

==========================================
Demographic models in species delimitation
==========================================

There are many ways to delimit species using genetic data, but one approach is to use genetic data to infer a demogrpahic model. 
Demographic models include information about population divergences, population size changes, and gene flow between populations.
Knowing this information about your focal group may help you to arrive at more biologically meaningful species delimitations `Smith and Carstens, 2020. <https://doi.org/10.1111/evo.13878>`_ 

========================================
Default demographic models in delimitpy
========================================

Given some user-input, delimitpy will create some default models that may be useful for delimiting species.
These models will incorporate divergence between populations, gene flow upon secondary contact between present-day populations,
and divergence with gene flow between sister populations.

To generate these models, the user must provide delimitpy with a `configuration file. <https://github.com/SmithLabBio/delimitpy/blob/main/config.txt>`_::

    [Model]
    species tree file = tree.nex # path to a species tree in nexus format
    migration matrix = migration.txt # path to a migration matrix
    symmetric = True # True if migration rates should always be symmetric, and only symmetric migration events should be included.
    secondary contact = True # True if you wish to consider secondary contact models.
    divergence with gene flow = True # True if you wish to consider divergence with gene flow models.
    max migration events = 2 # Maximum number of migration events to consider in one model.
    migration rate = U(1e-3, 1e-2) # Prior from which to draw migration rates. Only uniform priors are supported at present.

    [Other]
    output directory = test # Directory for storing output (should exist)
    seed = 1234 # random seed
    replicates = 10 # Number of replicates to simulate per model

