##############################
Building models with delimitpy
##############################

==========================================
Demographic models in species delimitation
==========================================

There are many ways to delimit species using genetic data, but one approach is to use genetic data to infer a demogrpahic model. 
Demographic models include information about population divergences, population size changes, and gene flow between populations.
Knowing this information about your focal group may help you to arrive at more biologically meaningful species delimitations `Smith and Carstens, 2020 <https://doi.org/10.1111/evo.13878>`_ 

========================================
Default demographic models in delimitpy
========================================

Given some user-input, delimitpy will create some default models that may be useful for delimiting species.
These models will incorporate divergence between populations, gene flow upon secondary contact between present-day populations,
and divergence with gene flow between sister populations.

To generate these models, the user must provide delimitpy with a configuration file.
.. literalinclude:: config.txt
   :text:
