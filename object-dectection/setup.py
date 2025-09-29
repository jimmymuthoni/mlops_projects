from setuptools import find_packages, setup

with open('requirements.txt') as file:
    requirements = file.read().splitlines()

setup(
    name="custom-object-detection",
    version="0.01",
    author="Jimmy",
    packages=find_packages(),
    install_requires=requirements
)