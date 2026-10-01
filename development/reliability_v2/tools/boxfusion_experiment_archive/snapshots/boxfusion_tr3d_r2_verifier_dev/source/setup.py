import glob
import os
import platform
import shutil
import sys
import warnings
from os import path as osp
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup


if __name__ == '__main__':
    setup(
        name='boxfusion',
        version='0.0.1',
        description=("Public release of Box Fusion"),
        url='https://github.com/lanlan96/BoxFusion',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False)
