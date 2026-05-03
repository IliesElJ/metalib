from setuptools import setup, find_packages

setup(
    name='metalib',
    version='0.1',
    author='Griffith Rees',
    packages=find_packages(),
    include_package_data=True,
    package_data={'metalib': ['styles/*.mplstyle']},
)
