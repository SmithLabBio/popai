##############################
Tutorial 1b: Processing data from a VCF
##############################

This tutorial demonstrates how to run popai when input data are in vcf format.

==========================================
Step 1: Prepare Input
==========================================

Example input data is available `here <https://github.com/SmithLabBio/popai/tree/main/tutorial_data/tutorial_1_data>`_.

You should have these in the directory you cloned when installing popai, in the subfolder tutorial_data.

Create a directory in which to run the tutorial, and copy these data to that directory::

    mkdir tutorial_1b
    cd tutorial_1b
    cp -r /path/to/downloaded/data/tutorial_1_data ./

For this tutorial, we will use the vcf data instead of the fasta files. There are two major differences:

1) Most notably, the data we will use are stored as a `vcf <https://github.com/SmithLabBio/popai/tree/main/tutorial_data/tutorial_1_data/alignments/alignment.vcf>`_.
2) The population file we will use is called populations_vcf.txt. 
    The populations file is different than the one used for the fasta files. This is because in fasta files, we should have one sequence per chromosome. The vcf files we take as input are from diploid organisms, and they are formatted such that each organism has an ID, rather than each chromosome. 
    Because of this, the popfile has one observation per organism, rather than one observation per chromosome.

You will need to slighly modify the config.txt file.

1) Change the entry for 'alignments' to None.
2) Add a line:
    vcf = ./tutorial_1_data/alignments/alignment.vcf 
3) Change the popfile to ./tutorial_1_data/populations_vcf.txt

==========================================
Step 2: Proceed with steps 3-6 of tutorial 1A.
==========================================
