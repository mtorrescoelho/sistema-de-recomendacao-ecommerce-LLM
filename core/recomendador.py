"""
Sistema Central de Recomendação
===============================

Este módulo centraliza a lógica de recomendação, integrando:
1. Personalização semântica e comportamental (personalizacao.py)
2. Refinamento por popularidade, sazonalidade e padrões globais (refinamento.py)

Ideal para servir recomendações de livros altamente relevantes e adaptadas ao utilizador.
"""

from core.personalizacao import SistemaPersonalizacao
from core.refinamento import SistemaRefinamento

class Recomendador:
    def __init__(self, dados):
        self.personalizacao = SistemaPersonalizacao()
        self.refinamento = SistemaRefinamento()
        self.dados = dados

    async def inicializar(self):
        await self.personalizacao.inicializar(self.dados)
        await self.refinamento.inicializar(self.dados)

    async def recomendar(self, consulta, usuario_id=None, top_k=10, categoria_filtro=None, preco_max=None):
        # 1. Personalização (busca semântica, histórico, etc)
        resultados_personalizados = await self.personalizacao.recomendar(
            consulta=consulta,
            usuario_id=usuario_id,
            top_k=top_k,
            categoria_filtro=categoria_filtro,
            preco_max=preco_max
        )
        # 2. Refinamento (popularidade, sazonalidade, etc)
        resultados_refinados = await self.refinamento.refinar_resultados(
            resultados_personalizados,
            self.personalizacao.padroes_categoria,
            self.personalizacao.padroes_sazonalidade
        )
        return resultados_refinados[:top_k]

    async def processar_feedback(self, usuario_id, produto_id, tipo_feedback, avaliacao=None, comentario=None):
        return await self.personalizacao.processar_feedback(
            usuario_id=usuario_id,
            produto_id=produto_id,
            tipo_feedback=tipo_feedback,
            avaliacao=avaliacao,
            comentario=comentario
        )

    async def obter_historico_usuario(self, usuario_id, limit=20):
        return await self.personalizacao.obter_historico_usuario(usuario_id, limit)

    async def obter_estatisticas(self):
        # Podes combinar estatísticas de personalização e refinamento se quiseres
        return await self.personalizacao.obter_estatisticas()

    def status_modelo(self):
        return self.personalizacao.status_modelo()