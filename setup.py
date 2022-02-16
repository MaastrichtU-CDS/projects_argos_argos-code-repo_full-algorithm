from os import path
from codecs import open
from setuptools import setup, find_packages

# get current directory
here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# # read the API version from disk
# with open(path.join(here, 'vantage6', 'tools', 'VERSION')) as fp:
#     __version__ = fp.read()

# setup the package
setup(
    name='argosfeddeep',
    version="1.0.0",
    description='argosfeddeep',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AnanyaCN/argos-docker-token2',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'nibabel==3.2.1',
        'numpy==1.19.5',
        'tensorflow==2.4.3',
        'pyjwt==1.7.1',
        'tensorboard==2.6.0',
        'tensorboard-data-server==0.6.1',
        'tensorboard-plugin-wit==1.8.0',
        'tensorflow==2.4.3',
        'tensorflow-estimator==2.4.0',
        'h5py==2.10.0',
        'nibabel==3.2.1',
        'numpy==1.19.5',
        'opencv-python==4.5.3.56',
        'scikit-image',
        'scipy==1.5.4',
        'flask==2.0.2'
    ])
