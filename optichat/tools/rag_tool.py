from loguru import logger
from typing import Dict, Any, List
from typing import Optional
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from openai import embeddings
from optichat.config.constants import CFG
from optichat.config.rag_cfg import *


def init_code_rag(path: str, model_name: str):
    """
path -> docs -> update vector store collection 
Recommended way to load source code: 
https://docs.langchain.com/oss/python/integrations/document_loaders/source_code
The parser can be disabled for small files.
This approach needs path to be a folder (or a single file) and uses glob pattern to load files.
TODO: Not sure if there exists an alternative way to load files from a list of specific files.
    """
    code_loader = GenericLoader.from_filesystem(
        path,
        glob="*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=CODE_RAG_PARSER_THRESHOLD))
    docs = code_loader.load()

    if CODE_RAG_IS_SPLITTED:
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=CODE_RAG_CHUNK_SIZE, chunk_overlap=CODE_RAG_CHUNK_OVERLAP)
        docs = code_splitter.split_documents(docs)

    collection_name = f"{model_name}_code"
    init_chroma_collection(collection_name, docs, empty_existing=True)


def init_paper_rag(paths: List[str], model_name: str):
    """
    path -> docs -> update vector store collection 
    """
    docs = []
    for path in paths:
        if path.endswith(".pdf"):
            pdf_loader = PyPDFLoader(path)
            docs.extend(pdf_loader.load())
        elif path.endswith(".txt"):
            txt_loader = TextLoader(path)
            docs.extend(txt_loader.load())
        else:
            raise ValueError(f"File type of {path} not supported.")

    if PAPER_RAG_IS_SPLITTED:
        paper_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PAPER_RAG_CHUNK_SIZE, chunk_overlap=PAPER_RAG_CHUNK_OVERLAP, add_start_index=True)
        docs = paper_splitter.split_documents(docs)

    collection_name = f"{model_name}_paper"
    init_chroma_collection(collection_name, docs, empty_existing=True)


def init_chroma_collection(collection_name, docs, empty_existing):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist_directory = PERSIST_DIRECTORY
    if empty_existing:
        # TODO: for development only, delete the collection first and then add documents to re-build the collection again
        vector_store = get_chroma_vs(collection_name, embeddings, persist_directory)
        logger.debug(f"Deleting collection {[col.name for col in vector_store._client.list_collections()]} first for re-building.")
        vector_store._client.delete_collection(name=collection_name)
        logger.debug(f"After deletion, existing collections are {[col.name for col in vector_store._client.list_collections()]}")
    vector_store = get_chroma_vs(collection_name, embeddings, persist_directory)
    logger.debug(f"Building collection {collection_name} by adding documents...")
    vector_store.add_documents(documents=docs)
    logger.debug(f"After adding documents, existing collections are {[col.name for col in vector_store._client.list_collections()]}")


def get_chroma_vs(collection_name, embeddings, persist_directory):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory)
    return vector_store


def code_rag(request: str, tool_context: ToolContext) -> str:
    """
    code_rag retrieves code blocks from <models_code> by
    performing semantic similarity search against the submitted request.

    Args:
        request: the information that you want to get from <models_code>, be specific and detailed

    Returns:
        string containing the code blocks that are relevant to the request submitted by you.
    """
    cfg = tool_context.state[CFG]
    if "models_code" in cfg:
        model_name = cfg["model_name"]
        collection_name = f"{model_name}_code"
        vector_store = get_chroma_vs(collection_name=collection_name,
                                     embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
                                     persist_directory=PERSIST_DIRECTORY)
        retriever = vector_store.as_retriever(search_type=CODE_RAG_SEARCH_TYPE,
                                              search_kwargs=CODE_RAG_SEARCH_KWARGS)
        result = retriever.invoke(request)
        return {"result": result}
    else:
        return {"result": "No 'models_code' in cfg, cannot use code_rag."}
    

def paper_rag(request: str, tool_context: ToolContext) -> str:
    """
    paper_rag retrieves paper contents from <models_paper> by
    performing semantic similarity search against the submitted request.

    Args:
        request: the information that you want to get from <models_paper>, be specific and detailed

    Returns:
        string containing the paper contents that are relevant to the request submitted by you.
    """
    cfg = tool_context.state[CFG]
    if "models_paper" in cfg:
        model_name = cfg["model_name"]
        collection_name = f"{model_name}_paper"
        vector_store = get_chroma_vs(collection_name=collection_name,
                                     embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
                                     persist_directory=PERSIST_DIRECTORY)
        retriever = vector_store.as_retriever(search_type=PAPER_RAG_SEARCH_TYPE,
                                              search_kwargs=PAPER_RAG_SEARCH_KWARGS)
        result = retriever.invoke(request)
        return {"result": result}
    else:
        return {"result": "No 'models_paper' in cfg, cannot use paper_rag."}