## Usage

Since `molfeat-hype` is a molfeat plugin, it follows the same integration principle as with any other molfeat plugin. 

The following shows examples of how to use the `molfeat-hype` plugin package automatically when installed.


1. Using directly this package

```python

from molfeat_hype.trans.llm_embeddings import LLMTransformer

mol_transf =  LLMTransformer(kind="sentence-transformers/all-mpnet-base-v2")
```

2. enable autodiscovery as plugin in molfeat, and addition of all embedding classes as importable attribute to the entry point group `molfeat.trans.pretrained`

```python
# put this somewhere in you code (e.g in the root __init__ file)
# plugins should be any subword of 'molfeat_hype' 
from molfeat.plugins import load_registered_plugins
load_registered_plugins(add_submodules=True, plugins=["hype"])
```

```python
# this is now possible everywhere
from molfeat.trans.pretrained import LLMTransformer
mol_transf =  LLMTransformer(kind="sentence-transformers/all-mpnet-base-v2")
```

Once you have your molecule transformer defined, just use it like any `molfeat` `MoleculeTransformer`


```python
import datamol as dm
smiles = dm.freesolv()["smiles"].values[:5]
mol_transf(smiles)
```

### Caching
We heavily encourage you to set a cache for computing your molecular representation. By default a temporary `in-memory` cache will be set and will be purged when the python runtime is stopped.

### OpenAI

If you are using the OpenAI embeddings, you need to either provide a 'open_ai_key' argument or define one as an environment variable 'OPEN_AI_KEY'.

Please refer to OpenAI's [API keys documentation](https://platform.openai.com/account/api-keys) to setup your API keys. You will need to activate the billing.


### llama.cpp

Llama integration requires the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) which is a python binding for the [llama-cpp](https://github.com/ggerganov/llama.cpp) package. You will also need the Llama model weights.

**Do not share any IPFS, magnet links, or any other links to model downloads anywhere in this repository, including in issues, discussions or pull requests**. They will be immediately deleted.

### Warning

This repository contains an experimental investigation of LLM embeddings for molecules. Please note that the consistency and usefulness of the returned molecular embeddings are not guaranteed. This project is meant for fun and exploratory purposes only, and should not be used as a demonstration of LLM capabilities for molecular embeddings. Any statements made in this repository are the opinions of the authors and do not necessarily reflect the views of any affiliated organizations or individuals. Use at your own risk.