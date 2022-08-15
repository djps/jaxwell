import setuptools

setuptools.setup(
    name='jaxwell',
    #version=read("jaxwell", "VERSION"),
    version='0.1',
    description='3D iterative FDFD electromagnetic solver.',
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    #author='Jesse Lu',
    #author_email='mr.jesselu@gmail.com',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    #install_requires=["numpy>=1.18.5", "jax>=0.2.4"],
    install_requires=read_requirements(".requirements/requirements.txt"),
    url='https://github.com/stanfordnqp/jaxwell',
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
