
# :comet: `molfeat-hype`

[![PyPI](https://img.shields.io/pypi/v/molfeat-hype)](https://pypi.org/project/molfeat-hype/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/molfeat-hype)](https://pypi.org/project/molfeat-hype/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/maclandrol/molfeat-hype/blob/main/LICENSE)
[![test](https://github.com/maclandrol/molfeat-hype/actions/workflows/test.yml/badge.svg)](https://github.com/maclandrol/molfeat-hype/actions/workflows/test.yml)
[![code-check](https://github.com/maclandrol/molfeat-hype/actions/workflows/code-check.yml/badge.svg)](https://github.com/maclandrol/molfeat-hype/actions/workflows/code-check.yml)
[![release](https://github.com/maclandrol/molfeat-hype/actions/workflows/release.yml/badge.svg)](https://github.com/maclandrol/molfeat-hype/actions/workflows/release.yml)

## Overview

`molfeat-hype` is an extension to `molfeat` that investigates the performance of embeddings from various LLMs trained without explicit molecular context, for molecular modelling. It leverages some of the most hyped LLM models in NLP to answer the following question:

```Is there even any need to pretrain/finetune LLMs on molecular context to get good molecular representations ?```

If you want an answer to this question, see the [benchmarks](./tutorials/benchmark.ipynb)

Two class of LLM embeddings are supported in `molfeat-hype`:

1. **Classic Embeddings**

    These are classical embeddings provided by foundation models or any LLMs. The models available here includes `OpenAI's text-embedding-ada-002` model, `llama` and several embedding models accessible through [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers/tree/master).

2. **Intruction-based Embeddings**
   
   These are models that have been trained to follow instruction (thus act like ChatGPT) or are conditional models that requires a prompt.

   - Chat-GPT (`openai/gpt-3.5-turbo`):  Asking ChatGPT to provide his best embeddings for molecules.
   - [alpaca](https://github.com/tatsu-lab/stanford_alpaca)
    
    In addition to those models, you can also provide the quantized weights of any ChatGPT-like models. For example, [GPT4All](https://github.com/nomic-ai/gpt4all) models can be supported through the `InstructPretrainedTransformer`.

## Installation

You can install `molfeat-hype` with:

```bash
mamba install -c conda-forge molfeat-hype
```
or 

```bash
pip install molfeat-hype
```

`molfeat-hype` depends mostly on [molfeat](https://github.com/datamol-io/molfeat) and [langchain](https://github.com/hwchase17/langchain).  For a list of complete dependencies, please see the [env.yml](./env.yml) file.


### Attribution 

- To learn more about [`molfeat`](https://github.com/datamol-io/molfeat), please visit https://molfeat.datamol.io/. 

- To learn more about the plugin system of molfeat, please see [extending molfeat](https://molfeat-docs.datamol.io/stable/developers/create-plugin.html)

- Please refer to the [`langchain`](https://github.com/hwchase17/langchain) documentation for any question related to langchain.


## Contributing

As an open source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infra, or better documentation.

For detailed information on how to contribute, see our [contribution guide](./contribute.md).


### Disclaimer
This repository contains an experimental investigation of LLM embeddings for molecules. Please note that the consistency and usefulness of the returned molecular embeddings are not guaranteed. This project is meant for fun and exploratory purposes only, and should not be used as a demonstration of LLM capabilities for molecular embeddings. Any statements made in this repository are the opinions of the authors and do not necessarily reflect the views of any affiliated organizations or individuals. Use at your own risk.

## License

Under the Apache-2.0 license. See [LICENSE](LICENSE).
