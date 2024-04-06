##############################
An Explanation of the Demographic Models
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

To generate these models, the user must provide delimitpy with a `configuration file <https://github.com/SmithLabBio/delimitpy/blob/main/tutorial_data/config.txt>`_.

In the configuration file, the user provides delimitpy with several pieces of information that determine how the model set is built.

* **species tree file**: The species tree (or trees) in this file are used to determine the relationships among populations in models (i.e., **the topology**). The species tree file also contains nodal annotations which tell delimitpy which priors to use for **population sizes** and **divergence times**.
* **migration matrix**: The migration matrix in this file is used by delimitpy to decide which populations may experience migration.
* **symmetric**: This determines whether delimitpy considers only symmetric migration events or also considers asymmetric migration.
* **secondary contact**: This determines whether delimitpy considers secondary contact models. If true, delimitpy will consider models for which migration begins half-way between the most recent divergence event and the present and ends in the present.
* **divergence with gene flow**: This determines whether delimitpy considers divergence with gene flow models, in which gene flow begins immediately after divergence and ends half way between the species divergence and the next divergence event (or the present if there are no more divergence events.)
* **max migration events**: This determines the maximum number of migration events considered in any single model and is helpful for limiting model space.
* **migration rate**: The prior from which migration rates are drawn.
* **constant Ne**: If True, delimitpy will use the same population sizes for all populations in the model.

Using this information, delimitpy will build a default model set. 

Importantly, whether using the CLI or running functions from delimitpy in python directly, users can check these models visually and ensure that they look as desired. When using the CLI, the user can run the simulate_data command with the flags --plot and without the flag --simulate to only plot the models. When running in python, the user can use the validate_models function of the ModelBuilder class.

