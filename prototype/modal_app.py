"""Modal deployment for Team Kinyeti — Agricultural Extension RAG prototype.

Hosts the Gradio web interface on Modal with serverless GPU inference.
Auto-scales to zero when idle (conserving free credits), with zero visitor limits.
"""

from pathlib import Path
import modal

PROTOTYPE_DIR = Path(__file__).resolve().parent

DENSE_MODEL_ID = "Overwatch886/bge-base-agri"
RERANKER_MODEL_ID = "Overwatch886/bge-reranker-large-agri"


MODEL_BUILD_TAG = "v2-retrained-dense"


def download_models():
    """Bake fine-tuned model weights into the container image during build."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"[{MODEL_BUILD_TAG}] Downloading latest model weights from Hugging Face...")
    SentenceTransformer("Overwatch886/bge-base-agri", revision="main")
    CrossEncoder("Overwatch886/bge-reranker-large-agri", revision="main")


# Build the container image with all dependencies and pre-cached model weights
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio>=4.44",
        "fastapi[standard]",
        "sentence-transformers>=3.0",
        "torch",
        "pandas",
        "rank-bm25",
        "huggingface_hub",
    )
    .env(
        {
            "DENSE_MODEL_ID": DENSE_MODEL_ID,
            "RERANKER_MODEL_ID": RERANKER_MODEL_ID,
            "DOCUMENTS_CSV": "/root/prototype/data/documents.csv",
        }
    )
    .run_function(download_models)
    .add_local_dir(str(PROTOTYPE_DIR), remote_path="/root/prototype")
)

app = modal.App("team-kinyeti-rag")


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=300,  # keep container warm for 5 minutes after requests
    timeout=600,
    max_containers=1,  # Gradio requires sticky sessions to avoid multi-container queue disconnects
)
@modal.asgi_app()
def ui():
    import os
    import sys

    os.chdir("/root/prototype")
    sys.path.insert(0, "/root/prototype")

    import app as gradio_app

    if gradio_app.PIPELINE is None:
        gradio_app.build_pipeline()
    demo = gradio_app.build_interface()

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()
    return mount_gradio_app(web_app, demo, path="/")
