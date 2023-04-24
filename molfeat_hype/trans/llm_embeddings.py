from typing import Union
from typing import List
from typing import Optional

import os
import dotenv
import datamol as dm
import platformdirs
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings as LangChainEmbeddings
from molfeat.trans.pretrained.base import PretrainedMolTransformer

dotenv.load_dotenv()
CACHE_DIR = str(platformdirs.user_cache_dir(appname="molfeat_hype"))


class LLMEmbeddingsTransformer(PretrainedMolTransformer):
    """
    Large Language Model Embeddings Transformer for molecule. This transformer embeds molecules using available LLM through langchain.
    Note that the LLMs do not have any molecular context as they were not trained on molecules or any specific molecular task.
    They are just trained on a large corpus of text.

    !!! warning "Caching computation"
        LLMs can be computationally expensive and even financially expensive if you use the OpenAI embeddings.
        To avoid recomputing the embeddings for the same molecules, we recommend using a molfeat Cache object.
        By default an in-memory  cache (DataCache) is used, but please explore other caching systems.

    ??? note "Using OpenAI Embeddings"
        If you are using the OpenAI embeddings, you need to either provide a 'open_ai_key'
        argument or define one as an environment variable 'OPEN_AI_KEY'.
        Please note that only the `text-embedding-ada-002` model is supported.
        Please refer to OPENAI's documentation for more information.

    ??? note "Using LLAMA Embeddings"
        The Llama embeddings are provided via the the python bindings of `llama.cpp`.
        We do not provide the path to the quantized llama model. However it's easy to find and them online,
        someone said there is a torrent/IPFS/direct download somewhere or llama weight, then you can quantized them yourself.
        When the model weight are provided but are

    ??? note "Using Sentence Transformer Embeddings"
        The sentence transformer embeddings are based on the SentenceTransformers package

    """

    SUPPORTED_EMBEDDINGS = [
        "openai/text-embedding-ada-002",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "llama.cpp",
    ]

    def __init__(
        self,
        kind=Union[str, LangChainEmbeddings],
        standardize: bool = True,
        precompute_cache: bool = True,
        n_jobs: int = 0,
        dtype=float,
        openai_api_key: Optional[str] = None,
        random_seed: Optional[int] = None,
        quantized_model_path: Optional[str] = None,
        parallel_kwargs: Optional[dict] = None,
        **params,
    ):
        """Instantiate a LLM Embeddings transformer

        Args:
            kind: kind of LLM to use. Supported LLMs are accessible through the SUPPORTED_EMBEDDINGS attribute. Here are a few:
                - "openai/text-embedding-ada-002"
                - "sentence-transformers/all-MiniLM-L6-v2"
                - "sentence-transformers/all-mpnet-base-v2"
                - "llama.cpp"
            standardize: if True, standardize smiles before embedding
            precompute_cache: if True, add a cache to cache the embeddings for the same molecules.
            n_jobs: number of jobs to use for preprocessing smiles.
            dtype: data type to use for the embeddings return type
            openai_api_key: openai api key to use. If None, will try to get it from the environment variable OPENAI_API_KEY
            random_seed: random seed to use for reproducibility
            quantized_model_path: path to the quantized model for llama.cpp. If None, will try to get it from the environment variable quantized_MODEL_PATH
            **params: parameters to pass to the LLM embeddings. See langchain documentation
        """

        self.kind = kind
        self.model = None
        self.standardize = standardize
        self.random_seed = random_seed
        if isinstance(kind, str):
            if not kind.startswith(tuple(self.SUPPORTED_EMBEDDINGS)):
                raise ValueError(
                    f"Unknown LLM type {kind} requested. Supported models are {self.SUPPORTED_EMBEDDINGS}"
                )
            if kind.startswith("openai/"):
                if openai_api_key is None:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                self.model = OpenAIEmbeddings(
                    model=kind.replace("openai/", ""), openai_api_key=openai_api_key, **params
                )
            elif kind.startswith("sentence-transformers/"):
                self.model = HuggingFaceEmbeddings(
                    model_name=kind, model_kwargs=params, cache_folder=CACHE_DIR
                )
            else:
                if quantized_model_path is None:
                    quantized_model_path = os.environ.get("quantized_MODEL_PATH")
                if quantized_model_path is not None and dm.fs.exists(quantized_model_path):
                    self._create_symlink(quantized_model_path, kind)
                else:
                    model_base_name = os.path.splitext(os.path.basename(quantized_model_path))[0]
                    quantized_model_path = dm.fs.glob(dm.fs.join(CACHE_DIR, f"{model_base_name}*"))
                    if len(quantized_model_path) == 0:
                        raise ValueError(
                            f"Could not find the quantized model {model_base_name} anywhere, including in the cache dir {CACHE_DIR}"
                        )
                    quantized_model_path = quantized_model_path[0]
                self.model = LlamaCppEmbeddings(model_path=quantized_model_path, **params)

        super().__init__(
            precompute_cache=precompute_cache,
            n_jobs=n_jobs,
            dtype=dtype,
            device="cpu",
            parallel_kwargs=parallel_kwargs,
            **params,
        )

    def _create_symlink(self, model_path, model_name: Optional[str] = None):
        """Create model symlink to cache dir to ensure that the model is accessible next time.
        If the model_name is None, then the basename of model_path will be used

        Args:
            model_path: path to the model
            model_name: name of the model
        """
        if model_name is None:
            model_name = os.path.basename(model_path)
        sym_link_path = dm.fs.join(CACHE_DIR, model_name)
        if not os.path.exists(sym_link_path):
            os.symlink(model_path, sym_link_path)
        return sym_link_path

    def _convert(self, inputs: List[Union[str, dm.Mol]], **kwargs):
        """Convert the list of input molecules into the proper format for embeddings"""
        self._preload()

        if isinstance(inputs, (str, dm.Mol)):
            inputs = [inputs]

        def _to_smiles(x):
            out = dm.to_smiles(x) if not isinstance(x, str) else x
            if self.standardize:
                with dm.without_rdkit_log():
                    return dm.standardize_smiles(out)
            return out

        parallel_kwargs = getattr(self, "parallel_kwargs", {})

        if len(inputs) > 1:
            smiles = dm.utils.parallelized(
                _to_smiles,
                inputs,
                n_jobs=self.n_jobs,
                **parallel_kwargs,
            )
        else:
            smiles = [_to_smiles(x) for x in inputs]
        return smiles

    def _embed(self, smiles: List[str], **kwargs):
        """_embed takes a list of smiles or molecules and return the featurization
        corresponding to the inputs.  In `transform` and `_transform`, this function is
        called after calling `_convert`

        Args:
            smiles: input smiles
        """
        return self.model.embed_documents(smiles)
