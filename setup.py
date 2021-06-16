
import setuptools
from tools.req import parse_requirements

install_reqs = parse_requirements('./requirements.txt')
reqs = [str(ir) for ir in install_reqs]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ds-suite",
    version="1.0.0",
    author="Daniel Cavalli",
    author_email="daniel@cavalli.dev",
    description="A brute-force based way of fiding the best ratio for your data. Focused on Tree models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielcavalli/ds-suite",
    install_requires=reqs,
    packages=['ds-suite'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
