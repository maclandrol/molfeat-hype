# Contribute

Thank you for your interest in contributing to `molfeat-hype`! As part of the molfeat community, we welcome all contributions, including code, documentation, and other forms of support. However you choose to contribute, please be mindful and respect our [code of conduct](https://github.com/maclandrol/molfeat-hype/blob/main/.github/CODE_OF_CONDUCT.md).

## Ways to contribute

You can contribute to `molfeat-hype` in several ways:

- Fix existing code issues.
- Submit issues related to bugs or new features.
- Suggest or implement new featurizers.
- Improve existing documentation or add new tutorials.

For a more detailed description of the development lifecycle, please refer to the rest of this document.

## Setting up a development environment

To get started, fork and clone the repository, then install the dependencies. It is strongly recommended that you do so in a new conda environment:

```bash
mamba env create -n molfeat_hype -f env.yml
conda activate molfeat_hype
pip install -e .
```

## Continuous Integration

`molfeat-hype` uses Github Actions to:

- **Build and test** the code.
- **Check code formatting** using `black`.
- **Build and deploy documentation** for each commit and tag.

## Running tests

You can run the tests with:

```bash
pytest
```

## Building the documentation

To build and serve the documentation locally, run:

```bash
mkdocs serve
```

## Submitting pull requests

If you're considering a large code contribution, please open an issue first to get early feedback on the idea. Once you think the code is ready to be reviewed, push it to your fork and open a pull request. A code owner or maintainer will review your PR and give you feedback.
