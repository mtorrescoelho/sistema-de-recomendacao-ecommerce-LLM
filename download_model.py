# download_modelo.py
from huggingface_hub import snapshot_download

# Faz o download do modelo para a pasta "modelo-feedback"
snapshot_download("C:\Users\maria\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2", local_dir="modelo-feedback", force_download=True)
