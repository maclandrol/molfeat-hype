import unittest as ut
import pytest


class TestImport(ut.TestCase):
    def test_import_plugin(self):
        try:
            from molfeat_padel.calc.padel import PadelDescriptors as PD
        except ImportError:
            pytest.fail("Failed to import PadelDescriptors from molfeat_padel")

        from molfeat.plugins import load_registered_plugins

        load_registered_plugins()
        try:
            from molfeat.calc import PadelDescriptors
        except ImportError:
            pytest.fail("Failed to import PadelDescriptors from molfeat")

        assert PD == PadelDescriptors, "PadelDescriptors is not the same class"
