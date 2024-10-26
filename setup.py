from setuptools import setup, find_packages

setup(
    name='atanu_mlkit',
    version='0.0.2',
    author="Atanu Debnath",
    author_email="playatanu@gmail.com",
    description="Machine Learning Kit",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/playatanu/atanu-mlkit",
    project_urls={
        "Documentation": "https://playatanu.github.io/atanu-mlkit/",
        "Source": "https://github.com/playatanu/atanu-mlkit",
        "Tracker": "https://github.com/playatanu/atanu-mlkit/issues",
    },
    packages=find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[],
    python_requires='>=3.6',
)