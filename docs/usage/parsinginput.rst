##############################
Input Data
##############################

Below, we provide information on the required input data. As an example, we use the data in example/test1. delimitR requires inputs (listed below). We will discuss each of these in turn.

* Configuration file
* Empirical Data
    * Alignments
    * Pop file
* Model Building Information
    * Species Tree
    * Migration Matrix

========================================
Configuration File
========================================

The user must provide delimitpy with a `configuration file <https://github.com/SmithLabBio/delimitpy/blob/main/tutorial_data/config.txt>`_.::


    [Model]
    species tree file = ../../tree.nex # Path to a species tree in nexus format.
    migration matrix = ../../migration.txt # Path to a migration matrix
    symmetric = True # True if migration rates should always be symmetric, and only symmetric migration events should be included.
    secondary contact = True # True if you wish to consider secondary contact models.
    divergence with gene flow = False # True if you wish to consider divergence with gene flow models.
    max migration events = 2 # Maximum number of migration events to consider in one model.
    migration rate = U(1e-5, 1e-4) # Prior from which to draw migration rates. Only uniform priors are supported at present.

    [Other]
    output directory = example_results # Directory for storing output (should exist).
    seed = 1234 # Random seed.
    replicates = 10 # Number of replicates to simulate per model.

    [Simulations]
    mutation rate = U(1e-8, 1e-7) # Prior from which to draw mutation rates. Only uniform priors are supported at present.
    substitution model = JC69 # Substitution model to use in simulations.

    [Data]
    alignments = ../../examples/test1/alignments # Path to alignments
    popfile = ../../examples/test1/populations.txt # Path to popfile

========================================
Empirical Data
========================================

------------
Alignments
------------

The user will provide a path to the alignments in the configuration file (see below). These alignments must be in fasta format, and there must be one alignment per locus. 

An `example set of alignments <https://github.com/SmithLabBio/delimitpy/blob/main/examples/test1/alignments>`_ is provided on GitHub.

------------
Pop File
------------

The user will provide a path to a file assigning individuals to populations in the config file (see below). This file must consist of two columns, with the heading 'individual' and 'population'.

An `example population file <https://github.com/SmithLabBio/delimitpy/blob/main/examples/test1/populations.txt>`_ is provided on GitHub.

========================================
Model Building Information
========================================

------------
Species Tree
------------

The user must provide a path to a nexus file with a species tree. There are some specific requirements for the `species tree file <https://github.com/SmithLabBio/delimitpy/blob/main/examples/test1/tree.nex>`_::

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

The user must provide a path to a file with a `migration matrix <https://github.com/SmithLabBio/delimitpy/blob/main/examples/test1/migration.txt>`_ indicating whether migration is allowed between all pairs of lineages::

    ,A,B,C,AB,ABC
    A,T,T,T,T,T
    B,T,T,T,T,T
    C,T,T,T,T,T
    AB,T,T,T,T,T
    ABC,T,T,T,T,T

Note that T indicates that migration is allowed between two taxa, while F indicates that migration is not allowed. The elements along the diagonal will be ignored. Ancestral populations must be included.


