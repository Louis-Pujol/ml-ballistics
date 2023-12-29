from setuptools import setup, find_packages

setup(
    name='mlballistics',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    extras_require={
        'dev': [
            'pytest',
        ],
    },
)
