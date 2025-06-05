# api/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TipoFeedback(str, Enum):
    """Tipos de feedback possíveis"""
    CURTIU = "curtiu"
    NAO_CURTIU = "nao_curtiu"
    COMPROU = "comprou"
    SALVOU = "salvou"
    COMPARTILHOU = "compartilhou"

class RecomendacaoRequest(BaseModel):
    """Request para gerar recomendações"""
    consulta: str = Field(..., description="Texto da consulta do usuário")
    usuario_id: Optional[str] = Field(None, description="ID do usuário para personalização")
    top_k: int = Field(5, ge=1, le=20, description="Número de recomendações a retornar")
    categoria_filtro: Optional[str] = Field(None, description="Filtrar por categoria específica")
    preco_max: Optional[float] = Field(None, description="Preço máximo dos livros")
    incluir_explicacao: bool = Field(True, description="Incluir explicação da recomendação")

    class Config:
        json_schema_extra = {
            "example": {
                "consulta": "Procuro um livro de ficção científica sobre inteligência artificial",
                "usuario_id": "user123",
                "top_k": 5,
                "categoria_filtro": "Ficção Científica",
                "preco_max": 25.0,
                "incluir_explicacao": True
            }
        }

class RecomendacaoResponse(BaseModel):
    """Response com recomendação de livro"""
    produto_id: str = Field(..., description="ID único do produto")
    titulo: str = Field(..., description="Título do livro")
    categoria: str = Field(..., description="Categoria do livro")
    descricao: str = Field(..., description="Descrição do livro")
    preco: Optional[float] = Field(None, description="Preço do livro")
    avaliacao_media: Optional[float] = Field(None, description="Avaliação média (0-5)")
    numero_avaliacoes: int = Field(0, description="Número total de avaliações")
    relevancia_score: float = Field(..., description="Score de relevância (0-1)")
    explicacao: Optional[str] = Field(None, description="Explicação da recomendação")
    tags: List[str] = Field(default_factory=list, description="Tags associadas ao livro")
    
    class Config:
        json_schema_extra = {
            "example": {
                "produto_id": "liv_001",
                "titulo": "Neuromancer",
                "categoria": "Ficção Científica",
                "descricao": "Romance cyberpunk clássico sobre inteligência artificial...",
                "preco": 19.99,
                "avaliacao_media": 4.2,
                "numero_avaliacoes": 156,
                "relevancia_score": 0.87,
                "explicacao": "Recomendado por ser um clássico da ficção científica sobre IA",
                "tags": ["cyberpunk", "inteligencia-artificial", "futuro"]
            }
        }

class FeedbackRequest(BaseModel):
    """Request para registrar feedback"""
    usuario_id: str = Field(..., description="ID do usuário")
    produto_id: str = Field(..., description="ID do produto")
    tipo_feedback: TipoFeedback = Field(..., description="Tipo de feedback")
    avaliacao: Optional[int] = Field(None, ge=1, le=5, description="Avaliação de 1 a 5")
    comentario: Optional[str] = Field(None, max_length=500, description="Comentário opcional")
    contexto_recomendacao: Optional[str] = Field(None, description="Contexto da recomendação original")

    class Config:
        json_schema_extra = {
            "example": {
                "usuario_id": "user123",
                "produto_id": "liv_001",
                "tipo_feedback": "curtiu",
                "avaliacao": 4,
                "comentario": "Gostei muito da recomendação!",
                "contexto_recomendacao": "Procuro ficção científica"
            }
        }

class StatusResponse(BaseModel):
    """Response de status do sistema"""
    status: str = Field(..., description="Status atual do sistema")
    message: str = Field(..., description="Mensagem descritiva")
    version: str = Field(..., description="Versão da API")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da resposta")

class HistoricoItem(BaseModel):
    """Item do histórico do usuário"""
    timestamp: datetime = Field(..., description="Data/hora da interação")
    consulta: str = Field(..., description="Consulta original")
    produto_id: str = Field(..., description="ID do produto recomendado")
    titulo: str = Field(..., description="Título do livro")
    acao: str = Field(..., description="Ação realizada pelo usuário")
    relevancia_score: float = Field(..., description="Score de relevância na época")

class EstatisticasResponse(BaseModel):
    """Response com estatísticas do sistema"""
    total_usuarios: int = Field(..., description="Total de usuários únicos")
    total_recomendacoes: int = Field(..., description="Total de recomendações geradas")
    total_feedbacks: int = Field(..., description="Total de feedbacks recebidos")
    taxa_aceitacao: float = Field(..., description="Taxa de aceitação das recomendações")
    categorias_populares: List[Dict[str, Any]] = Field(..., description="Categorias mais populares")
    modelo_stats: Dict[str, Any] = Field(..., description="Estatísticas do modelo ML")
    performance_metrics: Dict[str, float] = Field(..., description="Métricas de performance")

class ErroResponse(BaseModel):
    """Response de erro padronizada"""
    erro: str = Field(..., description="Tipo de erro")
    mensagem: str = Field(..., description="Mensagem de erro")
    detalhes: Optional[Dict[str, Any]] = Field(None, description="Detalhes adicionais do erro")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp do erro")

# Modelos para configuração
class ConfiguracaoPersonalizacao(BaseModel):
    """Configurações de personalização"""
    peso_historico: float = Field(0.3, ge=0, le=1, description="Peso do histórico nas recomendações")
    peso_categoria_preferida: float = Field(0.2, ge=0, le=1, description="Peso das categorias preferidas")
    peso_avaliacao: float = Field(0.3, ge=0, le=1, description="Peso das avaliações")
    peso_similaridade: float = Field(0.2, ge=0, le=1, description="Peso da similaridade semântica")
    decay_temporal: float = Field(0.1, ge=0, le=1, description="Decaimento temporal das preferências")

class MetricasAvaliacao(BaseModel):
    """Métricas de avaliação do sistema"""
    precisao: float = Field(..., description="Precisão das recomendações")
    recall: float = Field(..., description="Recall das recomendações")
    f1_score: float = Field(..., description="F1-Score")
    ndcg: float = Field(..., description="Normalized Discounted Cumulative Gain")
    diversidade: float = Field(..., description="Diversidade das recomendações")
    novidade: float = Field(..., description="Novidade das recomendações")
    tempo_resposta_medio: float = Field(..., description="Tempo médio de resposta (ms)")