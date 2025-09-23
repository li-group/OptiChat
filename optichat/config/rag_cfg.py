EMBEDDING_MODEL = "text-embedding-3-large"
PERSIST_DIRECTORY = "./chroma_langchain_db"


# the minimum number of lines that the source code file must have to be segmented using the parser
CODE_RAG_PARSER_THRESHOLD = 100 
CODE_RAG_IS_SPLITTED = False
CODE_RAG_CHUNK_SIZE = 100
CODE_RAG_CHUNK_OVERLAP = 10
CODE_RAG_SEARCH_TYPE = "similarity"  # similarity”, “mmr”, or “similarity_score_threshold”
CODE_RAG_NUM_OF_RESULTS = 4
CODE_RAG_NUM_OF_FETCH_K = 10  # only for mmr
CODE_RAG_LAMBDA_MULT = 0.5  # only for mmr
CODE_RAG_SCORE_THRESHOLD = 0.8  # only for similarity_score_threshold
if CODE_RAG_SEARCH_TYPE == "similarity":
    CODE_RAG_SEARCH_KWARGS = {"k": CODE_RAG_NUM_OF_RESULTS}
elif CODE_RAG_SEARCH_TYPE == "mmr":
    CODE_RAG_SEARCH_KWARGS = {"k": CODE_RAG_NUM_OF_RESULTS, 
                              "fetch_k": CODE_RAG_NUM_OF_FETCH_K, 
                              "lambda_mult": CODE_RAG_LAMBDA_MULT}
elif CODE_RAG_SEARCH_TYPE == "similarity_score_threshold":
    CODE_RAG_SEARCH_KWARGS = {"k": CODE_RAG_NUM_OF_RESULTS, 
                              "score_threshold": CODE_RAG_SCORE_THRESHOLD}
else:
    raise ValueError(f"CODE_RAG_SEARCH_TYPE {CODE_RAG_SEARCH_TYPE} not supported.")


PAPER_RAG_IS_SPLITTED = False
PAPER_RAG_CHUNK_SIZE = 1000
PAPER_RAG_CHUNK_OVERLAP = 100
PAPER_RAG_SEARCH_TYPE = "similarity"  # similarity”, “mmr”, or “similarity_score_threshold”
PAPER_RAG_NUM_OF_RESULTS = 4
PAPER_RAG_NUM_OF_FETCH_K = 10  # only for mmr
PAPER_RAG_LAMBDA_MULT = 0.5  # only for mmr
PAPER_RAG_SCORE_THRESHOLD = 0.8  # only for similarity_score_threshold
if PAPER_RAG_SEARCH_TYPE == "similarity":
    PAPER_RAG_SEARCH_KWARGS = {"k": PAPER_RAG_NUM_OF_RESULTS}
elif PAPER_RAG_SEARCH_TYPE == "mmr":
    PAPER_RAG_SEARCH_KWARGS = {"k": PAPER_RAG_NUM_OF_RESULTS, 
                              "fetch_k": PAPER_RAG_NUM_OF_FETCH_K, 
                              "lambda_mult": PAPER_RAG_LAMBDA_MULT}
elif PAPER_RAG_SEARCH_TYPE == "similarity_score_threshold":
    PAPER_RAG_SEARCH_KWARGS = {"k": PAPER_RAG_NUM_OF_RESULTS, 
                              "score_threshold": PAPER_RAG_SCORE_THRESHOLD}
else:
    raise ValueError(f"PAPER_RAG_SEARCH_TYPE {PAPER_RAG_SEARCH_TYPE} not supported.")



