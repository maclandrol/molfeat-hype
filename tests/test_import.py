import pytest


class TestImport:
    def test_import_plugin(self):
        try:
            from molfeat_hype.trans.llm_embeddings import LLMTransformer as LLMT
        except ImportError:
            pytest.fail("Failed to import LLMTransformer from molfeat_hype")

        from molfeat.plugins import load_registered_plugins

        load_registered_plugins(add_submodules=True, plugins=["hype"])
        try:
            from molfeat.trans.pretrained import LLMTransformer
        except ImportError:
            pytest.fail("Failed to import LLMTransformer from molfeat")

        assert LLMT == LLMTransformer, "LLMTransformer is not the same class"
