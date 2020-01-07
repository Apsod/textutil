from setuptools import setup, find_packages

setup(
    name='textutil',
    version='0.0.1',
    packages=find_packages(),

    license='MIT',
    author='Amaru Cuba Gyllensten',
    entry_points = {
        'console_scripts': [
            'textutil=textutil.__main__:main',
        ]
    }
)
