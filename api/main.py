# api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.models import (
    RecomendacaoRequest, 
    RecomendacaoResponse, 
    FeedbackRequest,
    StatusResponse
)
from core.recomendador import RecomendadorMelhorado
from utils.data_loader import DataLoader
from utils.logger import setup_logger

# Configuração da aplicação
app = FastAPI(
    title="Sistema de Recomendação de Livros",
    description="API para recomendação personalizada de livros com aprendizado contínuo",
    version="1.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
logger = setup_logger(__name__)

# Instância global do recomendador (será inicializada no startup)
recomendador = None
data_loader = None

@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação"""
    global recomendador, data_loader
    try:
        logger.info("Inicializando sistema de recomendação...")
        
        # Carregar dados
        data_loader = DataLoader()
        dados = data_loader.carregar_todos_dados()
        
        # Inicializar recomendador
        recomendador = RecomendadorMelhorado(dados)
        await recomendador.inicializar()
        
        logger.info("Sistema inicializado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Limpeza ao fechar a aplicação"""
    logger.info("Encerrando sistema de recomendação...")

@app.get("/", response_model=StatusResponse)
async def root():
    """Endpoint de status da API"""
    return StatusResponse(
        status="online",
        message="Sistema de Recomendação de Livros está funcionando",
        version="1.0.0"
    )

@app.get("/health")
async def health_check():
    """Health check para monitoramento"""
    if recomendador is None:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    return {
        "status": "healthy",
        "components": {
            "recomendador": "ok",
            "data_loader": "ok",
            "modelo": recomendador.status_modelo()
        }
    }

@app.post("/recomendar", response_model=List[RecomendacaoResponse])
async def recomendar_livros(request: RecomendacaoRequest):
    """
    Gerar recomendações personalizadas de livros
    """
    if recomendador is None:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        logger.info(f"Gerando recomendações para: {request.consulta}")
        
        recomendacoes = await recomendador.recomendar(
            consulta=request.consulta,
            usuario_id=request.usuario_id,
            top_k=request.top_k,
            categoria_filtro=request.categoria_filtro,
            preco_max=request.preco_max
        )
        
        return recomendacoes
        
    except Exception as e:
        logger.error(f"Erro ao gerar recomendações: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/feedback")
async def registrar_feedback(request: FeedbackRequest):
    """
    Registrar feedback do usuário para melhorar recomendações futuras
    """
    if recomendador is None:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        logger.info(f"Registrando feedback: usuario={request.usuario_id}, produto={request.produto_id}")
        
        sucesso = await recomendador.processar_feedback(
            usuario_id=request.usuario_id,
            produto_id=request.produto_id,
            tipo_feedback=request.tipo_feedback,
            avaliacao=request.avaliacao,
            comentario=request.comentario
        )
        
        if sucesso:
            return {"status": "success", "message": "Feedback registrado com sucesso"}
        else:
            raise HTTPException(status_code=400, detail="Erro ao processar feedback")
            
    except Exception as e:
        logger.error(f"Erro ao registrar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/historico/{usuario_id}")
async def obter_historico(usuario_id: str, limit: int = 20):
    """
    Obter histórico de recomendações e interações do usuário
    """
    if recomendador is None:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        historico = await recomendador.obter_historico_usuario(usuario_id, limit)
        return historico
        
    except Exception as e:
        logger.error(f"Erro ao obter histórico: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/categorias")
async def listar_categorias():
    """
    Listar todas as categorias disponíveis
    """
    if data_loader is None:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        categorias = data_loader.obter_categorias()
        return {"categorias": categorias}
        
    except Exception as e:
        logger.error(f"Erro ao listar categorias: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/estatisticas")
async def obter_estatisticas():
    """
    Obter estatísticas do sistema de recomendação
    """
    if recomendador is None:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        stats = await recomendador.obter_estatisticas()
        return stats
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )