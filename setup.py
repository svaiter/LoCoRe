#! /usr/bin/env python
#
descr = """Low Complexity Regularizations toolbox in Python"""

import sys
import os
import shutil

DISTNAME = 'locore'
DESCRIPTION = 'Low Complexity Regularizations toolbox in Python'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Samuel Vaiter'
MAINTAINER_EMAIL = 'samuel.vaiter@ceremade.dauphine.fr'
URL = 'http://svaiter.github.com/locore'
LICENSE = 'MIT'
DOWNLOAD_URL = 'http://github.com/svaiter/locore/downloads'
VERSION = '0.1'

import setuptools  # we are using a setuptools namespace
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
        namespace_packages=['locore'])

    config.add_subpackage('locore')
    config.add_subpackage('locore/tests')

    return config

if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False, # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
             ]
    )
