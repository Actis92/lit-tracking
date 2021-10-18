import os

from setuptools import setup

package_root = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(package_root, "src/lit_tracking/version.py")) as fp:
    exec(fp.read(), version)
version = version["__version__"]

setup(
    name='lit-tracking',
    version=version,
    license='MIT',
    description='A python package for multiple object tracking using Pytorch Lightning',
    author='Luca Actis Grosso, Lucia Marinozzi',
    author_email='lucaactisgrosso@gmail.com, lucia.marinozzi91@gmail.com',
    url='https://github.com/Actis92/lit-tracking',
    download_url=f'https://github.com/Actis92/lit-tracking.git/archive/{version}.tar.gz',
    keywords=['PYTORCH LIGHTNING'],
    install_requires=['pytorch-lightning>=1.3.8,<2',
                      'torch>=1.4',
                      'hydra-core>=1.1.0'
                      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ]
)