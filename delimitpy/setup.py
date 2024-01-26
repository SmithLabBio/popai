from setuptools import setup, find_packages

setup(
    name='delimitpy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'dendropy',
        'pandas',
        'msprime',
        'numpy',
        'demes',
        'demesdraw',
        'matplotlib',
    ],
)
