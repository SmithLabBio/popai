from setuptools import setup, find_packages

setup(
    name='popai',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'dendropy>=4.6',
        'pandas>=1.5',
        'msprime>=1.3',
        'numpy>=1.26',
        'demes',
        'demesdraw',
        'matplotlib>3.6',
        'scikit-learn',
        'keras>3',
        'seaborn',
        'tabulate',
        'tensorflow>=2.18',
        'tqdm',
        'torch>=2'
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
