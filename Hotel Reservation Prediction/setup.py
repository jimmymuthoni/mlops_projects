from setuptools import setup, find_packages
from typing import List

def get_requirements()->List:
    requirements_list: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirements_list.append(requirement)
                    
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirements_list

setup(
    name = "Hotel Reservation Prediction",
    author = "jimmymuthoni",
    version = "0.1",
    packages = find_packages(),
    install_requires = get_requirements(),
)