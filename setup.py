from setuptools import setup, find_packages

setup(
    name='popai',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'dendropy',
        'pandas',
        'msprime',
        'numpy',
        'demes',
        'demesdraw',
        'matplotlib',
        'argparse',
        'scikit-learn',
        'keras',
        'seaborn',
        'tabulate',
        'tensorflow',
        'pandas',
        'pyslim',
        'tqdm',
        'torch'
    ],
    entry_points={
        'console_scripts': [
            'process_empirical_data=popai.cli_process_empirical_data:main',
            'simulate_data=popai.cli_simulate_data:main',
            'train_models=popai.cli_train_models:main',
            'apply_models=popai.cli_apply_models:main'
        ],
    }
)
