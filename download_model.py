from huggingface_hub import snapshot_download

snapshot_download("neuralmind/bert-base-portuguese-cased", force_download=True)
