from setuptools import setup, find_packages

setup(
    name="swarm-opt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
    ],
    author="SwarmOpt Maintainers",
    description="Swarm intelligence for deep learning hyperparameter tuning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/swarm-opt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
