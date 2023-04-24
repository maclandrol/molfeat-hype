import unittest as ut
import datamol as dm
import numpy as np

from molfeat.plugins import load_registered_plugins
from molfeat.trans import MoleculeTransformer

load_registered_plugins()
from molfeat.calc import PadelDescriptors


class TestPadel(ut.TestCase):
    r"""Test cases for descriptors and pharmacophore generation"""
    smiles = [
        "CCOc1c(OC)cc(CCN)cc1OC",
        "COc1cc(CCN)cc(OC)c1OC",
        "C[C@@H]([NH3+])Cc1c2ccoc2c(Br)c2ccoc12",
    ]
    EXTRA_LARGE_MOL = "CC(C)CC(NCCNC(=O)C(CCC(O)=O)NC(C)=O)C(=O)NC(Cc1ccc(O)cc1)C(=O)NC(CC(C)C)C(=O)NC(C(C)C)C(=O)NC(C)C(=O)NCC(=O)NC(CCC(O)=O)C(=O)NC(CCCNC(N)=N)C(=O)NCC(=O)NC(Cc1ccccc1)C(=O)NC(Cc1ccccc1)C(=O)NC(Cc1ccc(O)cc1)C(=O)NC(C(C)O)C(=O)N1CCCC1C(=O)NC(C)C(O)=O"

    def test_padel_calc(self):
        calc = PadelDescriptors()
        fps = calc(self.smiles[0])
        self.assertEqual(len(fps), len(calc))

    def test_padel_fp(self):
        smiles = dm.freesolv()["smiles"].values[:5]
        for feat in ["PadelDescriptors", PadelDescriptors()]:
            mol_transf = MoleculeTransformer(featurizer=feat, dtype="df")
            fps = mol_transf(smiles)
            self.assertEqual(fps.shape, (5, 2756))

        from molfeat_padel.calc.padel import PadelDescriptors as PD

        mol_transf2 = MoleculeTransformer(featurizer=PD(), dtype="df")
        fps2 = mol_transf2(smiles)
        np.testing.assert_allclose(fps.values, fps2.values, rtol=1e-05)


if __name__ == "__main__":
    ut.main()
