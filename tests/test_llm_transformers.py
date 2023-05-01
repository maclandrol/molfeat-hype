import pytest
import os
import datamol as dm

from molfeat.plugins import load_registered_plugins

load_registered_plugins(add_submodules=True, plugins=["hype"])

from molfeat.trans.pretrained import InstructLLMTransformer
from molfeat.trans.pretrained import LLMTransformer


@pytest.fixture
def openai_api_key():
    return os.environ.get("OPENAI_API_KEY", None)


# write the unit test for the molfeat-hype plugin that cover both
# the LLMTransformer and InstructLLMTransformer classes as well as some of their variants. you should have multiple test function in the test class


class TestLLMTransformers:
    smiles = dm.freesolv()["smiles"].values[:3]
    INSTRUCT_EMBEDDING_SIZE = 16

    def test_llm_classic(self, openai_api_key):
        embedder = LLMTransformer(
            kind="openai/text-embedding-ada-002", openai_api_key=openai_api_key
        )
        out = embedder(self.smiles)
        assert out.shape == (len(self.smiles), 1536)

        embedder = LLMTransformer(kind="sentence-transformers/all-mpnet-base-v2")
        out = embedder(self.smiles)
        assert out.shape == (len(self.smiles), 768)

    def test_llm_instruct(self, openai_api_key):
        embedder = InstructLLMTransformer(
            kind="openai/chatgpt",
            openai_api_key=openai_api_key,
            embedding_size=self.INSTRUCT_EMBEDDING_SIZE,
        )
        out = embedder(self.smiles)
        assert out.shape == (len(self.smiles), self.INSTRUCT_EMBEDDING_SIZE)
        embedder = InstructLLMTransformer(kind="hkunlp/instructor-base")
        out = embedder(self.smiles)
        assert out.shape == (len(self.smiles), 768)
