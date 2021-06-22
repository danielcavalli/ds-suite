
import setuptools

with open('./requirements.txt') as f:
    reqs = [line.rstrip() for line in f]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ds-suite",
    version="1.0.5",
    author="Daniel Cavalli",
    author_email="daniel@cavalli.dev",
    description="A brute-force based way of fiding the best ratio for your data. Focused on Tree models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielcavalli/ds-suite",
    install_requires=reqs,
    packages=['dssuite'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
