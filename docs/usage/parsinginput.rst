##############################
Input Data
##############################

Below, we provide information on the required input data. As an example, we use the data in tutorial_data. delimitR requires inputs (listed below). We will discuss each of these in turn.

* **Configuration File**
* Empirical Data
    * Alignments
    * Pop file
* Model Building Information
    * Species Tree
    * Migration Matrix

========================================
Configuration File
========================================

The user must provide popai with a `configuration file <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/config.txt>`_.::


    [Model]
    species tree file = ./popai/tutorial_data/tree.nex # Path to a species tree in nexus format.
    migration matrix = ./popai/tutorial_data/migration.txt # Path to a migration matrix
    symmetric = True # True if migration rates should always be symmetric, and only symmetric migration events should be included.
    secondary contact = True # True if you wish to consider secondary contact models.
    divergence with gene flow = False # True if you wish to consider divergence with gene flow models.
    max migration events = 1 # Maximum number of migration events to consider in one model.
    migration rate = U(1e-5, 1e-4) # Prior from which to draw migration rates. Only uniform priors are supported at present.
    constant Ne = True # population sizes equal across all populations

    [Other]
    seed = 1234 # Random seed.
    replicates = 1000 # Number of replicates to simulate per model.

    [Simulations]
    mutation rate = U(5e-9, 5e-8) # Prior from which to draw mutation rates. Only uniform priors are supported at present.
    substitution model = JC69 # Substitution model to use in simulations.

    [Data]
    alignments = ./popai/tutorial_data/alignments # Path to alignments
    popfile = ./popai/tutorial_data/populations.txt # Path to popfile

========================================
Empirical Data
========================================

------------
Alignments
------------

The user will provide a path to the alignments in the configuration file (see above). These alignments must be in fasta format, and there must be one alignment per locus. The files must end with .fa or .fasta.

An `example set of alignments <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/alignments>`_ is provided on GitHub.

------------
VCF
------------

As an alternative to a set of gene alignments, the user can provide a vcf. In the config file, set alignments = None, and add vcf = /path/to/input.vcf below.

Note: if your VCF does not contain information on contig lengths, you must supply the expected contig length via a "length" argument in the config file's Data section.

------------
Pop File
------------

The user will provide a path to a file assigning individuals to populations in the config file (see above). This file must consist of two columns, with the heading 'individual' and 'population'.

An `example population file <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/populations.txt>`_ is provided on GitHub.

If you use fasta files, then "individuals" should corrsespond to haplotypes. If you use a VCF, "individuals" should correspond to individuals.

========================================
Model Building Information
========================================

------------
Species Tree
------------

The user must provide a path to a nexus file with a species tree. There are some specific requirements for the `species tree file <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/tree.nex>`_::

    #NEXUS
    BEGIN TAXA;
          Dimensions NTax=3;
          TaxLabels A B C;
    END;

    BEGIN TREES;
          Tree species=((A[&ne=10000-50000],B[&ne=10000-50000])AB[&ne=10000-50000,div=10000-50000],C[&ne=10000-50000])ABC[&ne=10000-50000,div=100000-500000];
    END;

Requirements:

* Internal nodes must be labeled with names.

* For each leaf and internal node, include an annotation indicating the minimum and maximum values of the uniform distribution on the effective population size for the corresponding population.::

    [&ne=10000-50000]

* For each linternal node, include an annotation indicating the minimum and maximum values of the uniform distribution on the divergence time (in generations before the present)::

    [&div=10000-50000]

----------------
Migration Matrix
----------------

The user must provide a path to a file with a `migration matrix <https://github.com/SmithLabBio/popai/blob/main/tutorial_data/migration.txt>`_ indicating whether migration is allowed between all pairs of lineages::

    ,A,B,C,AB,ABC
    A,F,F,T,F,F
    B,F,F,F,F,F
    C,T,F,F,T,F
    AB,F,F,T,F,F
    ABC,F,F,F,F,F

Note that T indicates that migration is allowed between two taxa, while F indicates that migration is not allowed. The elements along the diagonal will be ignored. Ancestral populations must be included.


