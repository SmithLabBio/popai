#################################
User-specified demographic models
#################################


========================================
Configuration File
========================================

Instead of using the default model set, users can write their own models for use in popai. To do so, add a ``user models`` argument to the ``[Model]`` section of the config file. Its value is the path to a folder of model files::

    [Model]
    species tree file = None # Path to a species tree in nexus format.
    migration matrix = None # Path to a migration matrix
    symmetric = None # True if migration rates should always be symmetric, and only symmetric migration events should be included.
    secondary contact = None # True if you wish to consider secondary contact models.
    divergence with gene flow = None # True if you wish to consider divergence with gene flow models.
    max migration events = None # Maximum number of migration events to consider in one model.
    migration rate = None # Prior from which to draw migration rates. Only uniform priors are supported at present.
    constant Ne = None # population sizes equal across all populations
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

NOTE: When ``user models`` is set, the other settings in the ``[Model]`` section (the species tree, migration matrix, and so on) are not used, so they can be left as None. Everything about the models comes from the model files.

========================================
Models Folder
========================================

Inside the models folder, place one file per model. There are two naming requirements:

1. Model files must end in ``.model``. Other files in the folder are ignored.
2. Model file names must end in ``_x.model``, where x is a whole number, and each model needs its own number. The number becomes the model's label (``model_x``) in popai's output. The numbers should be sequential and begin with 0 (i.e., 0-4 for five models).

Anything can come before the number, so ``isolation_0.model`` and ``model_0.model`` both work, and both are labelled ``model_0``.

For example, my directory structure could look as follows:

- user_models/

  - model_0.model
  - model_1.model
  - model_2.model

========================================
Model Files
========================================

Model files contain up to three sections:

1. Populations (required)
2. Migration (optional): migration from the present day
3. Events (optional): splits, migration over a time window, and other changes

Anything after a ``#`` on a line is a comment.

------------------
Populations
------------------

In the Populations section, list every population in the following format::

    name=[min,max]

Notes:

* Name is the name of the population.
* Min and Max are the bounds of a uniform prior from which the population size is drawn.
* You must also include historical (ancestral) populations.
* Instead of a range, a size can be a fixed number (``A=20000``), the name of a population listed above it, to give the two the same size in every replicate (``B=A``), or an expression using sizes drawn above it (``AB=0.5*pop_C``).

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

The Migration section sets migration rates from the present day. It holds one table, with population names as the column headers and as the first entry of each row. Separate entries with spaces or tabs::

    [Migration]
    matrix =
            A	B	C
        A	0	[1e-5,1e-4]	0
        B	[1e-5,1e-4]	0	0
        C	0	0	0

Notes:

* The entry in row j, column k is the rate at which lineages move from population j to population k, backwards in time (msprime's M[j, k]). Forward in time, it is the fraction of population j made up of migrants from population k each generation.
* Each entry is 0 (no migration), a rate (``2e-5``), a uniform prior on the rate (``[1e-5,1e-4]``), or a number of migrants instead of a rate (``1Nm`` or ``[0.5Nm,2Nm]``). Nm is converted to a rate by dividing by the size of the row's population (m = Nm / N_j).
* Each entry is drawn separately, so the rates in the two directions can differ.
* The table only needs to include populations that migrate, and the diagonal must be 0.
* The table can only include populations that exist at the present. Migration stops when a population splits from its ancestor (backwards in time). For migration involving an ancestral population, use a migration event (below).
* If two populations merge at time zero (see `Models with different numbers of populations/species`_), use the name of the merged population in the table.

------------------
Events
------------------

The Events section is where users can specify historical events.

Event entries are formatted as follows::

    event_name = function{parameters}

The event_name is chosen by the user, and can be used in other events (e.g., to use that event's time for another event). The name of a split, popsize, popgrowth, bottleneck, or pulse event stands for its time.

The function is chosen from amongst seven event types: split, symmetric_migration, asymmetric_migration, popsize, popgrowth, bottleneck, and pulse.

Parameters depend on the event type (see below). Each value can be:

* a number (e.g., ``25000`` or ``1e-5``)
* a range for a uniform prior (e.g., ``[10000,50000]``), with the smaller value first
* an expression using other events' times and population sizes. Expressions can use addition (+), subtraction (-), multiplication (*), division (/), parentheses, and the functions min, max, mean, and median (e.g., ``max(div_AB, 45000)`` or ``mean(div_AB, 3*1e4)``). The present-day size of population A is ``pop_A``.

Events can be listed in any order. popai works out the order to process them in, so that each event comes after the events it refers to. If two events refer to each other in a loop, popai gives an error.

popai currently accepts seven event types:

1. split
~~~~~~~~~~~~~~~~~~

Split events are used to specify population divergences (mergers backwards in time). To specify a split::

    div_name = split{time=[mintime, maxtime], descendants=[list of descendant populations], ancestor=name of ancestral population}

For example, to specify an event in which populations 'A' and 'B' merge to form population 'AB' between 10,000 and 50,000 generations ago::

    div_AB = split{time=[10000,50000], descendants=[A,B], ancestor=AB}

Splits can also be specified using an integer as the time::

    div_AB = split{time=25000, descendants=[A,B], ancestor=AB}

Or using a function involving another event::

    div_ABC = split{time=max(div_AB, 45000), descendants=[AB,C], ancestor=ABC}

A split is never drawn younger than the splits that form its descendants. For example, if div_AB and div_ABC have overlapping ranges, div_ABC is drawn from the part of its range that is no younger than div_AB in that replicate. If that's impossible (e.g., div_ABC is fixed at a time younger than div_AB), popai gives an error.

2. symmetric migration
~~~~~~~~~~~~~~~~~~~~~~

Symmetric migration events turn on migration between two populations for a window of time in the past. To specify a symmetric migration::

    mig_AB = symmetric_migration{start=[minstarttime, maxstarttime], stop=[minstoptime, maxstoptime], populations=[list of two populations], rate=[minrate,maxrate]}

For example, to specify migration between populations A and B that begins between 1,000 and 5,000 generations ago and ends when the two populations merge (split forward in time)::

    mig_AB = symmetric_migration{start=[1000,5000], stop=div_AB, populations=[A,B], rate=[1e-5,1e-4]}

The rate can be fixed (``rate=1e-5``), a uniform prior (``rate=[1e-5,1e-4]``), or a number of migrants (``rate=1Nm`` or ``rate=[0.5Nm,2Nm]``). For symmetric migration, one value is drawn for both directions. As a rate, both directions get the same rate. As Nm, both directions get the same number of migrants, so each direction's rate is Nm divided by the size of the population the lineages leave (backwards in time).

Start and stop are in generations before the present, so start is the more recent of the two. In each replicate, popai draws them so that:

* the start falls while both populations exist (after the split that forms them, for ancestral populations, and before the split that ends them), and
* the stop comes after the start.

If no start time can meet these conditions (e.g., a fixed start that is after the two populations have merged), popai gives an error.

3. asymmetric migration
~~~~~~~~~~~~~~~~~~~~~~~

Asymmetric migration events turn on migration in one direction for a window of time in the past. To specify an asymmetric migration::

    mig_AB = asymmetric_migration{start=[minstarttime, maxstarttime], stop=[minstoptime, maxstoptime], source=population, dest=population, rate=[minrate,maxrate]}

Please remember that these models are coalescent models, so everything is backwards in time, including the direction of migration: lineages move from source to dest backwards in time, which means dest sends migrants to source forward in time.

For example, to specify migration from A to B backwards in time, beginning between 1,000 and 5,000 generations ago and ending at their divergence time::

    mig_AB = asymmetric_migration{start=[1000, 5000], stop=div_AB, source=A, dest=B, rate=[1e-5,1e-4]}

The rate can be given in the same ways as for symmetric migration. As Nm, the rate is Nm divided by the size of the source population. Start and stop are drawn as for symmetric migration.

4. popsize
~~~~~~~~~~~~~~~~~~

Popsize events specify a change in the population size. To specify a popsize event::

    popsize_A = popsize{time=[mintime, maxtime], population=popname, size=newsize}

For example, to change the size of population A between 500 and 700 generations ago::

    popsize_A = popsize{time=[500, 700], population=A, size=[1000,2000]}

To specify a population bottleneck, users can use the following approach::

    popsize_C = popsize{time=[10000, 50000], population=C, size=0.01*pop_C}
    popsize_C2 = popsize{time=popsize_C+1000, population=C, size=pop_C}

Here, at a time drawn from 10,000 to 50,000 generations ago, population C reduces to 0.01 * its present-day size. 1000 generations later, it returns to its initial size.

5. popgrowth
~~~~~~~~~~~~~~~~~~

Popgrowth events specify a change in the exponential growth rate of a population. To specify a popgrowth event::

    popgrowth_A = popgrowth{time=[mintime, maxtime], population=popname, rate=rate}

The rate can be fixed or a range, and can be negative. As in msprime, a positive rate means the population grows forward in time (so it was smaller further back in time), and a negative rate means it shrinks.

6. bottleneck
~~~~~~~~~~~~~~~~~~

Bottleneck events are instantaneous bottlenecks, in which each lineage in the population coalesces into a single ancestor with probability prop. To specify a bottleneck::

    bottleneck_A = bottleneck{time=[mintime, maxtime], population=popname, prop=[minprop, maxprop]}

For example::

    bottleneck_A = bottleneck{time=[500,700], population=A, prop=[0.1,0.2]}

7. pulse
~~~~~~~~~~~~~~~~~~

Pulse events are single-generation (mass) migrations. Backwards in time, each lineage in source moves to dest with probability prop. Forward in time, this means that dest contributes a proportion prop of source's ancestry at that time. To specify a pulse migration::

    pulse_AB = pulse{time=[mintime, maxtime], source=population, dest=population, prop=[minprop, maxprop]}

For example::

    pulse_AB = pulse{time=[500,700], source=A, dest=B, prop=0.1}

========================================================
Models with different numbers of populations/species
========================================================

To specify models with different numbers of populations or species, always begin with the number of populations in the present day (i.e., corresponding to your sampled populations).

Specify events in which populations merge at time zero to generate models without divergence between some populations.

For example, if my data include three populations: A, B, and C, but I want to model a scenario in which A and B are not distinct, I would do the following::

    merge_AB = split{time=0, descendants=[A,B], ancestor=AB}

Samples from A and B are then simulated as samples from AB. To include migration from the present for AB in the migration table, use the name AB (not A or B).

========================================
Checks
========================================

popai checks model files as it reads them, and stops with a message naming the model file and the problem. For example, it gives an error for:

* a range with its larger value first, or a value that can't be right: a negative time or migration rate, a size of zero or less, or a proportion outside 0 to 1
* a typo in a value or an expression, or a name that isn't an event or a population size
* an event type that doesn't exist, or an event missing a value it needs
* a model file name that doesn't end in ``_x.model``, or two files with the same number
* timing that's impossible, such as a split younger than its descendants' splits, or migration that starts after its populations have merged

Values given as expressions are only known once they are drawn, so they are checked in each replicate.

========================================
Plotting user-specified models
========================================

popai will plot user-specified models. **PLEASE** look at the plots, and ensure that the models are interpreted as you intended. It is challenging to think of all the varieties of things people could specify, so testing this functionality is a huge challenge. If your models don't look like you think they should, check that you formatted entries correctly, and contact me so that I can provide clarification and make any necessary changes to ensure this functionality is as useful as possible!

Some caveats: we use demes for plotting, and demes will not allow populations to merge at time zero. For models with divergences at time zero (e.g., models with fewer populations), we move those divergences back by a very small time (e.g., 1 generation) for plotting only, along with anything at time zero that involves the merged populations. Be sure to look at the time scales when plotting your models before assuming they are not correctly interpreted.

Some models can't be drawn at all. For example, demes can't draw bottleneck events. For these models, the plot is replaced with a page explaining why. This doesn't change the simulations.

========================================
Running popai with user-specified models
========================================

To run popai with user-specified models, follow the command line instructions. The only change will be to your input files (the configuration file, and the directory with your models.)

========================================
Examples
========================================

In the `example_models <https://github.com/SmithLabBio/popai/blob/main/example_models/>`_ directory, I have provided three example model files and visualizations for the three models.

* Model 0: Two populations in the present (A+B and C). We will have samples from A, B, and C in the populations file and in our empirical data, but we want to test whether these are a single population. There is no divergence between A and B. The ancestor of A and B diverged from C between 50000 and 100000 generations ago.
* Model 1: Three populations in the present (A, B, and C). A and B diverge 10000 to 20000 generations ago. There is present-day gene flow between A and B. The ancestor of A and B diverged from C between 50000 and 100000 generations ago.
* Model 2: One population in the present (A+B+C). There is no divergence between any of our present-day populations.