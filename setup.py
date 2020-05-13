from setuptools import setup

from os import path

project_folder = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(project_folder, 'README.md'), 'r') as f:
    long_description = f.read()

_version = __import__('fxpmath').__version__

setup(
    name='fxpmath',
    version=_version,
    author='francof2a',
    author_email='empty@empty.com',
    packages=['fxpmath'],
    description='A python library for fractional fixed-point (base 2) arithmetic and binary manipulation.',
    url='https://github.com/francof2a/fxpmath',
    download_url = 'https://github.com/francof2a/fxpmath/archive/{}.tar.gz'.format(_version),
    license='MIT',
    keywords=['fixed point', 'fractional', 'math', 'python', 'fxpmath', 'fxp', 'arithmetic', 'FPGA', 'DSP'],
    install_requires=['numpy'],

    long_description = long_description,
    long_description_content_type="text/markdown",

    classifiers=[
        'Development Status :: 4 - Beta',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",

    ]
)

