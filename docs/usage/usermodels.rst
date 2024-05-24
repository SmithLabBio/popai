##############################
User-specified demographic models
##############################


========================================
Configuration File
========================================

Instead of using the default model set, users can generate their own models for use in popai. To do so, the user must add an arugment to the models section of the config file. This argument should be the path to a directory with model files.::


    [Model]
    species tree file = ./popai/tutorial_data/tree.nex # Path to a species tree in nexus format.
    migration matrix = ./popai/tutorial_data/migration.txt # Path to a migration matrix
    symmetric = True # True if migration rates should always be symmetric, and only symmetric migration events should be included.
    secondary contact = True # True if you wish to consider secondary contact models.
    divergence with gene flow = False # True if you wish to consider divergence with gene flow models.
    max migration events = 1 # Maximum number of migration events to consider in one model.
    migration rate = U(1e-5, 1e-4) # Prior from which to draw migration rates. Only uniform priors are supported at present.
    constant Ne = True # population sizes equal across all populations
    user models = ./model/

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
Models Folder
========================================
Inside the models folder, the user should place their model files. There are a two naming requirements:

1. Model files must end in the suffix '.model'.
2. Model file names must end in '_x', where x is an integer. The integers should be sequential and begin with 0 (i.e., 0-4 for five models).

For example, my directory structure could look as follows:

- user_models/
    - model_0.model
    - model_1.model
    - model_2.model

========================================
Model Files
========================================

Model files contain three major sections:
1. Populations
2. Migration
3. Events

------------------
Populations
------------------
In the Populations section, the user should list their populations in the following format::
    name=[min,max]

Notes:
* Name is the name of the populations.

* Min and Max are the minimum and maximum bounds on a uniform prior from which population sizes will be drawn.

* You must also include historical populations.


Example::

    [Populations]	# list populations and min and maximum values for uniform population size priors [min,max]
    A=[10000,50000]
    B=[10000,50000]
    C=[10000,50000]
    AB=[10000,50000]
    ABC=[10000,50000]

------------------
Migration
------------------
The Migration section contains the migration rate to use at the beginning of the simulation (the present). 
Each population, including ancestral populations, must be present. Rates can either be set to 0, or a uniform prior can be provided in the format [min,max].

**Entries must be separated by tabs.**

Example::

    [Migration]	# migration matrix with migration rates entered as a uniform prior [min,max] M[j,k] is the rate at which lineages move from population j to population k in the coalescent process. J is row, K is column
    matrix	=	
            A	B	C	AB	ABC
        A	0	[1e-5,1e-4]	0	0	0
        B	[1e-5,1e-4]	0	0	0	0
        C	0	0	0	0	0
        AB	0	0	0	0	0
        ABC	0	0	0	0	0

------------------
Events
------------------
The events section is where users can specify historical events. 

Event entries are formated as follows::
    index=type	[mintime]	[maxtime]	[type specific parameters]

The mintime and maxtime are the prior for a uniform distribution on the timing of the event.

**Entries must be separated by tabs.**

popai currently accepts five event types:

1. split

Split events are used to specify population divergences. To specify a split::
    1=split	[mintime]	[maxtime]	[list of derived populations]	[ancestral population]

For example, to specify an event in which pouplations 'A' and 'B' merge to form population 'AB' between 10,000 and 50,000 generations ago::
    1=split	10000	50000	["A","B"]	AB

2. symmetric migration

Symmetric migration events specify a change in the migration rate between two populations at some time in the past. To specify a symmetric migration::
    2=symmetric migration	[mintime]	[maxtime]	[list of populations]	[rate]

Rate can either be [min,max] value for a uniform prior, or a single floating point value.

For example, to specify migration beginning between populations A and B bewteen 1,000 and 5,000 generations ago.::
    2=symmetric migration	1000	5000	["A","B"]	[1e-5,1e-4]

3. asymmetric migration 

Asymmetric migration events specify a change in the migration rate between two populations at some time in the past. To specify an asymmetric migration::
    2=asymmetric migration	[mintime]	[maxtime]	[source]	[dest]	[rate]

Please remember that these models are coalescent models, so everything is backwards in time, including the direction of migration.

Rate can either be [min,max] value for a uniform prior, or a single floating point value.

For example, to specify asymigration beginning from A to B backwards in time bewteen 1,000 and 5,000 generations ago.::
    2=asymmetric migration	1000	5000	A	B	[1e-5,1e-4]

4. popsize

Popsize events specify a change in the population size and/or a change in the growth rate for a population. To specify a popsize event::
    3=popsize	[mintime]	[maxtime]	[population]	[new size]	[growth rate]

New size can either be a uniform prior specified as [min,max], or 'None' to keep the current population sized (used when changing rate only).

Growth rate can either be a uniform prior specified as [min,max] or 'None' to keep the current growth rate (used when changing size only).

For example, to change the size of population A between 500 and 700 generations ago::
    3=popsize	500	700	A	[1000,2000]	None

5. bottleneck

Bottleneck events specify a population bottleneck. To specify a bottleneck event::
    4=bottleneck	[mintime]	[maxtime]	[population]	[proportion]

Proportion is the probability of each lineage coalescing in a single ancestor.

For example, to specify a bottleneck in population A between 500 and 700 generations ago::
    4=bottleneck	500	700	A	0.1

========================================
Models with different numbers of pouplations/species
========================================

To specify models with different numbers of populations or species, always begin with the number of populations in the present day (i.e., corresponding to your sampled populations.)

Specify events in which populations merge at time zero to generate models without divergence between some populations. 

For example, if my data include three populations: A, B, and C, but I want to model a scenario in which A and B are not distinct, I would do the following::
    1=split	0	0	["A","B"]	AB

========================================
Plotting user-specified models
========================================
popai will plot user-specified models. **PLEASE** look at the plots, and ensure that the models are interpreted as you intended. It is challening to think of all the varieties of things people could specify, so testing this functionality is a huge challenge. If your models don't look like you think they should, check that you formatted entries correctly, and contact me so that I can provide clarification and make any necessary changes to ensure this functionality is as useful as possible!

One caveat: we used demes for plotting, and demes will not allow events to happen at time zero, or multiple events to happen at the same time. For models with divergences or other events at time zero (e.g., models with fewer populations), we will add a very small time (e.g., 1 generation) for plotting only. Be sure to look at the time scales when plotting your models before assuming they are not correctly interpreted.

========================================
Running popai with user-specified models
========================================
To run popai with user specified models, follow the command line instructions. The only change will be to your input files (the configuration file, and the directory with your models.)

Instructions for running popai with user-specified models by importing modules in python are coming soon. If they still aren't here, and you need them, contact me!

========================================
Examples
========================================
In the `example_models <https://github.com/SmithLabBio/popai/blob/main/example_models/>`_ directory, I have provided three example model files and visualizations for the three models.

* Model 0: Two populations in the present (A+B and C). We will have samples from A, B, and C in the populations file and in our empirical data, but we want to test whether these are a single population. There is no divergence between A and B. The ancestor of A and B diverged from C between 50000 and 100000 generations ago.
* Model 1: Three populations in the present (A, B, and C). A and B diverge 10000 to 20000 generations ago. There is present-day gene flow beteen A and B. The ancestor of A and B diverged from C between 50000 and 100000 generations ago.
* Model 2: One population in the present (A+B+C). There is no divergence between any of our present-day populations.