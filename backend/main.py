from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.connection import init_pool, close_pool
from routers import parse, chunk, embed, search, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    # Warm-up: load the embedding model into memory at startup
    from services.embedder import get_embedder
    await get_embedder()
    yield
    await close_pool()


app = FastAPI(
    title="Enterprise RAG API",
    description="Document intelligence & Q&A powered by n8n + RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parse.router,  prefix="/parse_document",  tags=["Parsing"])
app.include_router(chunk.router,  prefix="/chunk_document",  tags=["Chunking"])
app.include_router(embed.router,  prefix="/embed_chunks",    tags=["Embedding"])
app.include_router(search.router, prefix="/search",          tags=["Retrieval"])
app.include_router(query.router,  prefix="/query",           tags=["Generation"])


@app.get("/health")
async def health():
    return {"status": "ok"}
