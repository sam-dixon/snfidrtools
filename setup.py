from setuptools import setup

setup(name='IDRTools',
      version='0.1',
      description='Handy tools for working with the SNf IDR',
      url='http://github.com/sam-dixon/IDRTools',
      author='Sam Dixon',
      author_email='sam.dixon@berkeley.edu',
      license='None',
      packages=['IDRTools'],
      install_requires=['numpy', 'matplotlib', 'IPython', 'sncosmo'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
