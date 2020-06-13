from setuptools import setup

setup(name='data-generators',
      version='0.1',
      description='Library for generating additional data for datasets.',
      url='https://github.com/Craven-Biostat-Lab/data-generators',
      author='Shrehit Goel',
      author_email='data-generators-issues@fire.fundersclub.com',
      license='BSD 3-Clause',
      packages=['data_generators'],
      install_requires=[
          'numpy',
          'sklearn',
          'pomegranate @ git+http://github.com/shrehit/pomegranate.git@master#egg=pomegranate',
      ])
