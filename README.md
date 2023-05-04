
# :comet: `molfeat-hype`

<div align="center">
    <img src="docs/assets/molfeat-hype-cover.svg" width="100%">
</div>
<p align="center">
    <b> ☄️ molfeat-hype - A molfeat plugin that leverages the most hyped LLM models in NLP for molecular featurization.</b> <br />
</p>
<p align="center">
  <a href="https://maclandrol.github.io/molfeat-hype/" target="_blank">
      Docs
  </a>
</p>

---

[![PyPI](https://img.shields.io/pypi/v/molfeat-hype)](https://pypi.org/project/molfeat-hype/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/molfeat-hype)](https://pypi.org/project/molfeat-hype/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/maclandrol/molfeat-hype/blob/main/LICENSE)
[![test](https://github.com/maclandrol/molfeat-hype/actions/workflows/test.yml/badge.svg)](https://github.com/maclandrol/molfeat-hype/actions/workflows/test.yml)
[![code-check](https://github.com/maclandrol/molfeat-hype/actions/workflows/code-check.yml/badge.svg)](https://github.com/maclandrol/molfeat-hype/actions/workflows/code-check.yml)
[![release](https://github.com/maclandrol/molfeat-hype/actions/workflows/release.yml/badge.svg)](https://github.com/maclandrol/molfeat-hype/actions/workflows/release.yml)

## Overview
`molfeat-hype` is an extension of `molfeat` that investigates the performance of embeddings from various LLMs trained without explicit molecular context for molecular modeling. It leverages some of the most hyped LLM models in NLP to answer the following question:

```Is it necessary to pretrain/finetune LLMs on molecular context to obtain good molecular representations?```

To find an answer to this question, check out the [benchmarks](tutorials/benchmark.ipynb).

<details>
 <summary>Spoilers</summary>
 <strong>YES!</strong> Understanding molecular context/structure/properties is key to building good molecular featurizers. 
</details>

### LLMs

`molfeat-hype` supports two types of LLM embeddings:

1. **Classic Embeddings**: These are classical embeddings provided by foundation models (or any LLMs). The models available in this tool include OpenAI's `openai/text-embedding-ada-002` model, `llama`, and several embedding models accessible through [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers/tree/master).

2. **Instruction-based Embeddings**: These are models that have been trained to follow instructions (thus acting like ChatGPT) or are conditional models that require a prompt.

   - **Prompt-based instruction:** A model (like Chat-GPT: `openai/gpt-3.5-turbo`) is asked to act like an all-knowing AI assistant for drug discovery and provide the best molecular representation for the input list of molecules. Here, we parse the representation from the Chat agent output.
   - **Conditional embeddings:** A model trained for conditional text embeddings that takes instruction as additional input. Here, the embedding is the model underlying representation of the molecule conditioned by the instructions it received. For more information, see this [instructor-embedding](https://github.com/HKUNLP/instructor-embedding).

## Installation

You can install `molfeat-hype` using pip. `conda` installation is planned soon.

```bash
pip install molfeat-hype
```

`molfeat-hype` mostly depends on [molfeat](https://github.com/datamol-io/molfeat) and [langchain](https://github.com/hwchase17/langchain). Please see the [env.yml](./env.yml) file for a complete list of dependencies.

### Acknowledgements 

Check out the following projects that made molfeat-hype possible:

- To learn more about [`molfeat`](https://github.com/datamol-io/molfeat), please visit https://molfeat.datamol.io/. To learn more about the plugin system of molfeat, please see [extending molfeat](https://molfeat-docs.datamol.io/stable/developers/create-plugin.html)

- Please refer to the [`langchain`](https://github.com/hwchase17/langchain) documentation for any questions related to langchain.

## Usage

Since `molfeat-hype` is a `molfeat` plugin, it follows the same integration principle as with any other `molfeat` plugin. 

The following shows examples of how to use the `molfeat-hype` plugin package automatically when installed.

1. Using this package directly:

```python

from molfeat_hype.trans.llm_embeddings import LLMTransformer

mol_transf = LLMTransformer(kind="sentence-transformers/all-mpnet-base-v2")
```

2. Enabling autodiscovery as a plugin in `molfeat`, and addition of all embedding classes as an importable attribute to the entry point group `molfeat.trans.pretrained`:

```python
# Put this somewhere in your code (e.g., in the root __init__ file).
# Plugins should include any subword of 'molfeat_hype'.
from molfeat.plugins import load_registered_plugins
load_registered_plugins(add_submodules=True, plugins=["hype"])
```

```python
# This is now possible everywhere.
from molfeat.trans.pretrained import LLMTransformer
mol_transf = LLMTransformer(kind="sentence-transformers/all-mpnet-base-v2")
```

Once you have defined your molecule transformer, use it like any `molfeat` `MoleculeTransformer`:

```python
import datamol as dm
smiles = dm.freesolv()["smiles"].values[:5]
mol_transf(smiles)
```


## Changelog
See the latest changelogs at [CHANGELOG.rst](./CHANGELOG.rst).

## Maintainers

- @maclandrol

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether in the form of new features, improved infrastructure, or better documentation. 
For detailed information on how to contribute, see our [contribution guide](./contribute.md).


## Disclaimer
This repository contains an experimental investigation of LLM embeddings for molecules. Please note that the consistency and usefulness of the returned molecular embeddings are not guaranteed. This project is meant for fun and exploratory purposes only and should not be used as a demonstration of LLM capabilities for molecular embeddings. Any statements made in this repository are the opinions of the authors and do not necessarily reflect the views of any affiliated organizations or individuals. Use at your own risk.

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE) for details.

