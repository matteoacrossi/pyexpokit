from setuptools import setup, find_packages

setup(
    name='pyexpokit',
    version='0.0.1',
    url='https://github.com/matteoacrossi/pyexpokit.git',
    author='Matteo Rossi',
    description="Python implementation of Expokit's expv",
    packages=find_packages(),
    install_requires=['scipy'],
)