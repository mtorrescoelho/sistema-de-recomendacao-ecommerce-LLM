"""
Script para fazer download do modelo de análise de sentimentos. 
Execute este script antes de executar o modelo principal.
"""

import os
import logging
from huggingface_hub import snapshot_download
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações
MODELO_REPOSITORIO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PASTA_LOCAL = "modelo-feedback"

def verificar_modelo_existe():
    """Verifica se o modelo já foi descarregado"""
    if os.path.exists(PASTA_LOCAL) and os.listdir(PASTA_LOCAL):
        logger.info(f"✓ Modelo já existe em '{PASTA_LOCAL}'")
        return True
    return False

def fazer_download_modelo():
    """Faz download do modelo do Hugging Face"""
    try:
        logger.info(f"Iniciando download do modelo: {MODELO_REPOSITORIO}")
        logger.info(f"Destino: {PASTA_LOCAL}")
        
        # Criar pasta se não existir
        Path(PASTA_LOCAL).mkdir(exist_ok=True)
        
        # Fazer download
        snapshot_download(
            repo_id=MODELO_REPOSITORIO,
            local_dir=PASTA_LOCAL,
            force_download=False,  # Não força se já existir
            resume_download=True   # Retoma download se interrompido
        )
        
        logger.info("✓ Download concluído com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro durante o download: {e}")
        return False

def main():
    """Função principal"""
    logger.info("=== DOWNLOAD DO MODELO ===")
    
    # Verificar se já existe
    if verificar_modelo_existe():
        resposta = input("Modelo já existe. Deseja fazer download novamente? (s/N): ")
        if resposta.lower() not in ['s', 'sim', 'y', 'yes']:
            logger.info("Download cancelado pelo utilizador")
            return
    
    # Fazer download
    sucesso = fazer_download_modelo()
    
    if sucesso:
        logger.info("=== DOWNLOAD CONCLUÍDO ===")
        logger.info(f"Pode agora executar o modelo principal")
    else:
        logger.error("=== DOWNLOAD FALHADO ===")
        logger.error("Verifique a ligação à internet e tente novamente")

if __name__ == "__main__":
    main()