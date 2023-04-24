# Contribute

`molfeat-hype` is part of the molfeat community and as such follows the same guideline! We appreciate everyone's contribution and welcome all. Apart from code contributions, there are various other ways to support the community such as answering questions, providing assistance to others, enhancing the documentation, giving a ⭐️ or sharing the project.

However you choose to contribute, please be mindful and respect our [code of conduct](https://github.com/maclandrol/molfeat-hype/blob/main/.github/CODE_OF_CONDUCT.md).


## Ways to contribute

You can contribute to `molfeat` and `molfeat-hype` in several ways:

- Fix existing code issues.
- Submit issues related to bugs or new features.
- Suggest or implement new featurizers.
- Improve existing documentation or add new tutorials. 

For a more detailed description of the development lifecycle, please refer to the rest of this document.


## Setup a dev environment

First you'll need to fork and clone the repository. Once you have a local copy, install the dependencies. 
It is strongly recommended that you do so in a new conda environment.


```bash
mamba env create -n molfeat_hype -f env.yml
conda activate molfeat_hype
pip install -e .
```


## Continuous Integration

molfeat uses Github Actions to:

- **Build and test** `molfeat_hype`.
- **Check code formating** the code: `black`.
- **Documentation**: build and deploy the documentation on `main` and for every new git tag.

## Run tests

```bash
pytest
```

## Build the documentation

You can build and serve the documentation locally with:

```bash
# Build and serve the doc
mkdocs serve
```

## Submitting Pull Requests

If you're considering a large code contribution, please open an issue first to get early feedback on the idea. Once you think the code is ready to be reviewed, push it to your fork and open a pull request. A code owner or maintainer will review your PR and give you feedbacks.

## Release a new version

- Run check: `rever check`.
- Bump and release new version: `rever VERSION_NUMBER`.
- Releasing a new version will do the following tasks in this order:
  - Update `AUTHORS.rst`.
  - Update `CHANGELOG.rst`.
  - Bump the version number in `setup.py` and `_version.py`.
  - Add a git tag.
  - Push the git tag.
  - Add a new release on the GH repo associated with the git tag.
