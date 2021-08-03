from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

setup(
    name='ppchallenge',
    version='0.0.1',
    description="Code for the PP Challenge",
    author='Tolga Dincer',
    author_email='tolgadincer@gmail.com',
    license='MIT',
    keywords='power trade optimization forecasting',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=requirements,
)
