##############################
Building models with delimitpy
##############################

==========================================
Demographic models in species delimitation
==========================================

There are many ways to delimit species using genetic data, but one approach is to use genetic data to infer a demographic model. 
Demographic models include information about population divergences, population size changes, and gene flow between populations.
Knowing this information about your focal group may help you to arrive at more biologically meaningful species delimitations `Smith and Carstens, 2020. <https://doi.org/10.1111/evo.13878>`_ 

========================================
Default demographic models in delimitpy
========================================

Given some user-input, delimitpy will create some default models that may be useful for delimiting species.
These models will incorporate divergence between populations, gene flow upon secondary contact between present-day populations,
and divergence with gene flow between sister populations.

To generate these models, the user must provide delimitpy with a `configuration file <https://github.com/SmithLabBio/delimitpy/blob/main/config.txt>`_.::

    [Model]
    species tree file = tree.nex # Path to a species tree in nexus format.
    migration matrix = migration.txt # Path to a migration matrix
    symmetric = True # True if migration rates should always be symmetric, and only symmetric migration events should be included.
    secondary contact = True # True if you wish to consider secondary contact models.
    divergence with gene flow = True # True if you wish to consider divergence with gene flow models.
    max migration events = 2 # Maximum number of migration events to consider in one model.
    migration rate = U(1e-3, 1e-2) # Prior from which to draw migration rates. Only uniform priors are supported at present.

    [Other]
    output directory = test # Directory for storing output (should exist).
    seed = 1234 # Random seed.
    replicates = 10 # Number of replicates to simulate per model.

------------
Species Tree
------------

The user must provide a path to a nexus file with a species tree. There are some specific requirements for the `species tree file <https://github.com/SmithLabBio/delimitpy/blob/main/tree.nex>`_::

    #NEXUS
    BEGIN TAXA;
        Dimensions NTax=3;
        TaxLabels A B C;
    END;

    BEGIN TREES;
        Tree species=((A[&ne=1000-10000],B[&ne=1000-10000])AB[&ne=1000-10000,div=1000-50000],C[&ne=1000-10000])ABC[&ne=1000-10000,div=10000-100000];
    END;

Requirements:

* Internal nodes must be labeled with names.

* For each leaf and internal node, include an annotation indicating the minimum and maximum values of the uniform distribution on the effective population size for the corresponding population.::

    [&ne=1000-1000]

* For each linternal node, include an annotation indicating the minimum and maximum values of the uniform distribution on the divergence time (in generations before the present)::

    [&div=10000-100000]

----------------
Migration Matrix
----------------

The user must provide a path to a file with a `migration matrix <https://github.com/SmithLabBio/delimitpy/blob/main/migration.txt>`_ indicating whether migration is allowed between all pairs of lineages::

    ,A,B,C,AB,ABC
    A,T,T,T,T,T
    B,T,T,T,T,T
    C,T,T,T,T,T
    AB,T,T,T,T,T
    ABC,T,T,T,T,T

Note that T indicates that migration is allowed between two taxa, while F indicates that migration is not allowed. The elements along the diagonal will be ignored. Ancestral populations must be included.


