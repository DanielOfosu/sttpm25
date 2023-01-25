import os

from setuptools import find_packages, setup

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="spatiotemporaltransformer",
    version="1.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    description="Library for the Spatiotemporal Transformer Model",
    author="Daniel Ofosu",
    author_email="daniel.ofosu@aalto.fi",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(),
)
