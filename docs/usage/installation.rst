###################
Installing with pip
###################

popai needs Python 3.10 or later. TensorFlow, which popai uses for its neural networks, supports Python 3.10 to 3.13, so use one of those. popai has been tested with Python 3.11.

We recommend installing popai in its own environment, so that the packages it installs don't change your other software. For example, with conda::

    conda create -n popai python=3.11
    conda activate popai

To install popai::

    git clone https://github.com/SmithLabBio/popai.git
    cd popai/
    pip install .

OR::

    pip install git+https://github.com/SmithLabBio/popai/

This installs the latest versions of the packages popai uses. To install the exact versions popai was tested with instead, run this in the popai folder before ``pip install .``::

    pip install -r requirements.txt