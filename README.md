# pyexpokit
Python port of Expokit library by R. B. Sidje, based on the paper [R. B. Sidje, ACM Trans. Math. Softw. 24, 130-156 (1998)](https://dx.doi.org/10.1145/285861.285868).

Link to the original implementation of Expokit: https://www.maths.uq.edu.au/expokit/.

# Installation
Install with pip

    pip install git+https://github.com/matteoacrossi/pyexpokit.git
    
or clone the repository and install with

    python setup.py install
    
# Usage

    from pyexpokit import expmv
    
    expmv(t, A, v)
    
