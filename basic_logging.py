import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='a')
    ]
)

app_logger = logging.getLogger("feedhope.app")
embedding_logger = logging.getLogger("feedhope.embeddings")
vectorstore_logger = logging.getLogger("feedhope.vectorstore")
llm_logger = logging.getLogger("feedhope.llm")
auth_logger = logging.getLogger("feedhope.auth")
rate_limit_logger = logging.getLogger("feedhope.ratelimit")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)