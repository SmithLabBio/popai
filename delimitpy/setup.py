from setuptools import setup, find_packages

setup(
    name='delimitpy',
    version='0.2',
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
        'sklearn',
        'keras',
        'seaborn',
        'tabulate'
    ],
    entry_points={
        'console_scripts': [
            'process_empirical_data=delimitpy.cli_process_empirical_data:main',
            'simulate_data=delimitpy.cli_simulate_data:main',
            'train_models=delimitpy.cli_train_models:main',
            'apply_models=delimitpy.cli_apply_models:main'
        ],
    }
)
