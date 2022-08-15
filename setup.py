import io
import os

import setuptools


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("jwave", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open( os.path.join(os.path.dirname(__file__), *paths), encoding=kwargs.get("encoding", "utf8"), ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    """gets requirements"""
    out = [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]
    print(out)
    return out


setuptools.setup(
    name='jaxwell',
    #version=read("jaxwell", "VERSION"),
    version='0.1',
    description='3D iterative FDFD electromagnetic solver.',
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    #author='Jesse Lu',
    #author_email='mr.jesselu@gmail.com',
    packages=setuptools.find_packages(exclude=["tests", ".github"]),
    python_requires='>=3.6',
    #install_requires=["numpy>=1.18.5", "jax>=0.2.4"],
    install_requires=read_requirements(".requirements/requirements.txt"),
    #url='https://github.com/stanfordnqp/jaxwell',
    license='GNU Lesser General Public License (LGPL)',
    keywords=["jax", "electro-magnetic", "curl-curl", "maxwell", "simulation", "ultrasound", "differentiable-programming"],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Physics",],
)
