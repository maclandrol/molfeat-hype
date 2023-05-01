from typing import Union
from typing import List
from typing import Optional

import os
import platformdirs
import datamol as dm

CACHE_DIR = str(platformdirs.user_cache_dir(appname="molfeat_hype"))


def create_symlink(model_path, model_name: Optional[str] = None, cache_dir: str = CACHE_DIR):
    """Create model symlink to cache dir to ensure that the model is accessible next time.
    If the model_name is None, then the basename of model_path will be used

    Args:
        model_path: path to the model
        model_name: name of the model
        cache_dir: cache dir to use
    """
    model_out = os.path.basename(model_path)
    if model_name is not None:
        model_out = f"{model_name}/{model_out}"
    sym_link_path = dm.fs.join(cache_dir, model_out)
    dirname = os.path.dirname(sym_link_path)
    os.makedirs(dirname, exist_ok=True)
    if not os.path.exists(sym_link_path):
        os.symlink(model_path, sym_link_path)
    return sym_link_path


def convert_smiles(
    inputs: List[Union[str, dm.Mol]], parallel_kwargs: dict, standardize: bool = False
):
    """Convert the list of input molecules into the proper format for embeddings
    Args:
        inputs: list of input molecules
        parallel_kwargs: kwargs for datamol parallelization
        standardize: whether to standardize the smiles
    """

    if isinstance(inputs, (str, dm.Mol)):
        inputs = [inputs]

    def _to_smiles(x):
        out = dm.to_smiles(x) if not isinstance(x, str) else x
        if standardize:
            with dm.without_rdkit_log():
                return dm.standardize_smiles(out)
        return out

    if len(inputs) > 1:
        smiles = dm.utils.parallelized(
            _to_smiles,
            inputs,
            **parallel_kwargs,
        )
    else:
        smiles = [_to_smiles(x) for x in inputs]
    return smiles
