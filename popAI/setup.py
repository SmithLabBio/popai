from setuptools import setup, find_packages

setup(
    name='popAI',
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
        'pyslim'
    ],
    entry_points={
        'console_scripts': [
            'process_empirical_data=popAI.cli_process_empirical_data:main',
            'simulate_data=popAI.cli_simulate_data:main',
            'train_models=popAI.cli_train_models:main',
            'apply_models=popAI.cli_apply_models:main'
        ],
    }
)
