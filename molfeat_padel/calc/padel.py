from typing import Union
from typing import List
from typing import Optional
from typing import Dict
from typing import Any

import datamol as dm
import numpy as np
import padelpy

from functools import partial
from molfeat.utils.commons import requires_standardization
from molfeat.utils.commons import _clean_mol_for_descriptors
from molfeat.utils.datatype import to_numpy
from molfeat.calc.base import SerializableCalculator


class PadelDescriptors(SerializableCalculator):
    """
    Compute PaDEL descriptors for a molecule using padelpy (see https://github.com/ecrl/padelpy)
    For the original PaDEL descriptor software, see: http://www.yapcwsoft.com/dd/padeldescriptor/
    PaDEL descriptors are under the GNU AGPL license v3.0.
    The descriptor calculator does not mask errors in featurization and will propagate them.

    !!! note:
        PaDEL Descriptors can be very slow to compute, as an exception, this calculator supports batch computation.
        It's recommended to process smiles in batch for large datasets and save them to disk.

    """

    def __init__(
        self,
        descriptors: bool = True,
        fingerprints: bool = True,
        timeout: int = 30,
        replace_nan: bool = False,
        do_not_standardize: bool = False,
    ):
        """
        Padel descriptor computation
        Args:
            descriptors: Whether to compute descriptors
            fingerprints: Whether to compute fingerprints
            timeout: Timeout for the computation
            do_not_standardize: Whether to force standardize molecules or keep it the same
        """
        self.descriptors = descriptors
        self.fingerprints = fingerprints
        self.timeout = timeout
        self.replace_nan = replace_nan
        self.do_not_standardize = do_not_standardize
        self._length = None
        self._columns = None
        self.__init_meta_data()

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["replace_nan"] = self.replace_nan
        state["fingerprints"] = self.fingerprints
        state["timeout"] = self.timeout
        state["descriptors"] = self.descriptors
        state["do_not_standardize"] = getattr(self, "do_not_standardize", False)
        return state

    def __setstate__(self, state: dict):
        """Reload the class from pickling."""
        self.__dict__.update(state)
        self.__init_meta_data()

    def __init_meta_data(self):
        """
        Initialize the padel descriptors meta data for the current object
        """
        toy_smiles = "CCCC"
        out = padelpy.from_smiles(
            toy_smiles,
            fingerprints=self.fingerprints,
            descriptors=self.descriptors,
            timeout=self.timeout,
        )
        self._length = len(out)
        self._columns = [f"PaDEL_{name}" for name in out.keys()]

    @property
    def columns(self):
        """
        Get the name of all the descriptors of this calculator
        """
        return self._columns

    def __len__(self):
        """Return the length of the calculator"""
        return self._length

    @staticmethod
    def __cast_to_float(x):
        """Cast as value to float otherwise return None
        Args:
            x: The value to cast
        """
        try:
            x = float(x)
        except ValueError:
            x = float("nan")
        return x

    @requires_standardization(disconnect_metals=True, remove_salt=True)
    def __call__(self, mol: Union[dm.Mol, str]):
        r"""
        Get rdkit basic descriptors for a molecule

        Args:
            mol: the molecule of interest

        Returns:
            props (np.ndarray): list of computed mordred molecular descriptors
        """
        if isinstance(mol, (str, dm.Mol)):
            mol = [mol]
        smiles = [dm.to_smiles(dm.to_mol(m)) for m in mol]
        vals = padelpy.from_smiles(
            smiles,
            fingerprints=self.fingerprints,
            descriptors=self.descriptors,
            timeout=self.timeout,
        )
        vals = to_numpy([[self.__cast_to_float(x) for x in v.values()] for v in vals])
        if self.replace_nan:
            vals = np.nan_to_num(vals)
        if len(smiles) == 1:
            return vals[0]
        return vals

    def batch_compute(
        self,
        mols: Union[List[dm.Mol], List[str]],
        disconnect_metals: bool = True,
        remove_salt: bool = True,
        **parallel_kwargs,
    ):
        """Batch compute the padel descriptors
        Args:
            mols: List of molecules
            disconnect_metals: Whether to disconnect metals
            remove_salt: Whether to remove salt
            parallel_kwargs: Additional arguments for parallelization
        """
        if parallel_kwargs is None:
            parallel_kwargs = {}
        with dm.without_rdkit_log():
            mols = dm.parallelized(
                partial(
                    _clean_mol_for_descriptors,
                    disconnect_metals=disconnect_metals,
                    remove_salt=remove_salt,
                ),
                mols,
                **parallel_kwargs,
            )
            smiles = dm.parallelized(dm.to_smiles, mols, **parallel_kwargs)
        vals = padelpy.from_smiles(
            smiles,
            fingerprints=self.fingerprints,
            descriptors=self.descriptors,
            timeout=self.timeout,
        )
        vals = to_numpy([[self.__cast_to_float(x) for x in v.values()] for v in vals])
        if self.replace_nan:
            vals = np.nan_to_num(vals)
        return vals
