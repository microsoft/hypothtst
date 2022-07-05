from setuptools import setup, find_packages

setup(name='hypothtst',
      version='0.0.12',
      url='https://github.com/microsoft/hypothtst',
      license='MIT',
      author='Rohit Pandey',
      author_email='rohitpandey576@gmail.com', 'ropandey@microsoft.com'
      description='Add static script_dir() method to Path',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)
