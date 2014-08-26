# LoCoRe - Low Complexity Regularizations toolbox in Python

## Description

LoCoRe is a Python module dedicated to low complexity regularizations, such as sparse regularizations or low rank priors.
It includes procedures to compute solutions of these regularizations and identifiability criterions.

## Installation

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use

    $ python setup.py install --user

To install for all users on Unix/Linux

    $ python setup.py build

    $ sudo python setup.py install

## Test

After installation, you can launch the test suite from outside the
source directory (you will need to have the ``nose`` package installed)

    $ nosetests -v locore

## License

LoCoRe is a Python module distributed under the MIT license.
See COPYING for more info.
