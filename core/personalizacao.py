# core/recomendador.py
"""
Sistema Avançado de Personalização de Recomendações
===================================================

Este módulo implementa:
1. Geração de recomendações personalizadas com embeddings semânticos e histórico do utilizador
2. Integração de personalização baseada em preferências, feedback e padrões comportamentais
3. Suporte a explicações, tags e estatísticas detalhadas para cada recomendação
4. Otimização contínua do sistema com cache, análise de padrões e feedback do utilizador

Ideal para oferecer recomendações altamente relevantes e adaptadas a cada utilizador.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Optional, Any, Tuple
import asyncio
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import json
import pickle
from pathlib import Path

from api.models import RecomendacaoResponse, TipoFeedback
from core.personalizacao import SistemaPersonalizacao
from core.refinamento import SistemaRefinamento
from utils.text_processor import TextProcessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SistemaPersonalizacao:
    def __init__(self):
        model_dir = "caminho/para/resultados"
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def prever_sentimento(self, texto):
        inputs = self.bert_tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            probs = outputs.logits.softmax(dim=1)
            classe = probs.argmax(dim=1).item()
        return classe  # 0=Negativo, 1=Neutro, 2=Positivo

class RecomendadorMelhorado:
    
    def __init__(self, dados: Dict[str, pd.DataFrame]):
        self.dados = dados
        self.livros_df = dados['livros']
        self.feedbacks_df = dados['feedbacks']
        self.analise_dados = dados.get('analise', {})
        
        # Componentes do sistema
        # Inicialização dos sistemas de personalização e refinamento
        self.modelo = None
        self.text_processor = TextProcessor()
        self.personalizacao = SistemaPersonalizacao()
        self.refinamento = SistemaRefinamento()
        
        # Cache para embeddings
        self.embeddings_cache = {}
        self.cache_path = Path("cache/embeddings.pkl")
        self.cache_path.parent.mkdir(exist_ok=True)
        
        # Histórico de usuários
        self.historico_usuarios = defaultdict(list)
        self.perfis_usuarios = {}
        
        # Estatísticas
        self.estatisticas = {
            'total_recomendacoes': 0,
            'total_feedbacks': 0,
            'taxa_aceitacao': 0.0,
            'tempo_resposta': []
        }
        
        self.logger = logging.getLogger(__name__)
        
    async def inicializar(self):
        """Inicialização assíncrona do sistema"""
        try:
            self.logger.info("Carregando modelo de embeddings...")
            #carrega o modelo para gerar embeddings semânticos
            self.modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            self.logger.info("Carregando cache de embeddings...")
            await self._carregar_cache_embeddings()
            
            self.logger.info("Preparando embeddings dos livros...")
            await self._preparar_embeddings_livros()
            
            self.logger.info("Inicializando sistema de personalização...")
            await self.personalizacao.inicializar(self.dados)
            
            self.logger.info("Inicializando sistema de refinamento...")
            await self.refinamento.inicializar(self.dados)
            
            self.logger.info("Analisando padrões comportamentais...")
            await self._analisar_padroes_comportamentais()
            
            self.logger.info("Recomendador inicializado com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {str(e)}")
            raise
    
    async def _carregar_cache_embeddings(self):
        """Carregar cache de embeddings se existir"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                self.logger.info(f"Cache carregado: {len(self.embeddings_cache)} embeddings")
            except Exception as e:
                self.logger.warning(f"Erro ao carregar cache: {str(e)}")
                self.embeddings_cache = {}
    
    async def _salvar_cache_embeddings(self):
        """Salvar cache de embeddings"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            self.logger.warning(f"Erro ao salvar cache: {str(e)}")
    
    async def _preparar_embeddings_livros(self):
        """Preparar embeddings dos livros com cache"""
        if 'livros_embeddings' not in self.embeddings_cache:
            self.logger.info("Gerando embeddings dos livros...")
            
            # Criar textos descritivos melhorados
            textos_livros = []
            for _, livro in self.livros_df.iterrows():
                texto = self.text_processor.criar_texto_descritivo(livro)
                textos_livros.append(texto)
            
            # Gerar embeddings
            embeddings = self.modelo.encode(textos_livros, convert_to_tensor=True)
            
            self.embeddings_cache['livros_embeddings'] = embeddings
            self.embeddings_cache['livros_textos'] = textos_livros
            
            await self._salvar_cache_embeddings()
        
        self.logger.info(f"Embeddings preparados para {len(self.embeddings_cache['livros_textos'])} livros")
    
    async def _analisar_padroes_comportamentais(self):
        """Analisar padrões comportamentais dos dados históricos"""
        try:
            # Análise de padrões de compra por categoria
            if 'categorias_populares' in self.analise_dados:
                categorias_pop = pd.read_csv(self.analise_dados['categorias_populares'])
                self.padroes_categoria = categorias_pop.set_index('Categoria')['Frequencia'].to_dict()
            
            # Análise de sazonalidade
            if 'sazonalidade' in self.analise_dados:
                sazonalidade = pd.read_csv(self.analise_dados['sazonalidade'])
                self.padroes_sazonalidade = sazonalidade.to_dict('records')
            
            # Análise de preços
            if 'valor_medio_por_compras' in self.analise_dados:
                precos = pd.read_csv(self.analise_dados['valor_medio_por_compras'])
                self.padroes_preco = precos.to_dict('records')
                
        except Exception as e:
            self.logger.warning(f"Erro na análise de padrões: {str(e)}")
            self.padroes_categoria = {}
            self.padroes_sazonalidade = []
            self.padroes_preco = []
    
    async def recomendar(
        self,
        consulta: str,
        usuario_id: Optional[str] = None,
        top_k: int = 5,
        categoria_filtro: Optional[str] = None,
        preco_max: Optional[float] = None
    ) -> List[RecomendacaoResponse]:
        """Gerar recomendações personalizadas"""
        inicio = datetime.now()
        
        try:
            # 1. Processamento da consulta
            consulta_processada = self.text_processor.processar_consulta(consulta)
            
            # 2. Detecção de categoria melhorada
            categoria_detectada = await self._detectar_categoria_avancada(consulta_processada)
            categoria_filtro = categoria_filtro or categoria_detectada
            
            # 3. Filtragem inicial dos livros
            livros_candidatos = await self._filtrar_livros_candidatos(
                categoria_filtro, preco_max
            )
            
            # 4. Busca semântica
            resultados_semanticos = await self._busca_semantica(
                consulta_processada, livros_candidatos, top_k * 2
            )
            
            # 5. Personalização baseada no usuário
            if usuario_id:
                resultados_personalizados = await self.personalizacao.personalizar_resultados(
                    usuario_id, resultados_semanticos, consulta_processada
                )
            else:
                resultados_personalizados = resultados_semanticos
            
            # 6. Refinamento baseado em padrões comportamentais
            resultados_refinados = await self.refinamento.refinar_resultados(
                resultados_personalizados, self.padroes_categoria, self.padroes_sazonalidade
            )
            
            # 7. Diversificação e ranking final
            recomendacoes_finais = await self._diversificar_e_rankear(
                resultados_refinados, top_k, consulta_processada
            )
            
            # 8. Gerar explicações
            recomendacoes_com_explicacao = await self._gerar_explicacoes(
                recomendacoes_finais, consulta_processada, usuario_id
            )
            
            # 9. Registrar interação
            await self._registrar_interacao(usuario_id, consulta, recomendacoes_com_explicacao)
            
            # 10. Atualizar estatísticas
            tempo_resposta = (datetime.now() - inicio).total_seconds() * 1000
            self.estatisticas['tempo_resposta'].append(tempo_resposta)
            self.estatisticas['total_recomendacoes'] += 1
            
            return recomendacoes_com_explicacao
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar recomendações: {str(e)}")
            raise
    
    async def _detectar_categoria_avancada(self, consulta: str) -> Optional[str]:
        """Detecção avançada de categoria usando similaridade semântica"""
        categorias_disponiveis = self.livros_df["Categoria"].unique()
        
        # Criar embeddings das categorias se não existirem
        if 'categorias_embeddings' not in self.embeddings_cache:
            categoria_embeddings = self.modelo.encode(categorias_disponiveis, convert_to_tensor=True)
            self.embeddings_cache['categorias_embeddings'] = categoria_embeddings
        
        # Buscar categoria mais similar
        consulta_embedding = self.modelo.encode(consulta, convert_to_tensor=True)
        similaridades = util.semantic_search(
            consulta_embedding, 
            self.embeddings_cache['categorias_embeddings'], 
            top_k=1
        )[0]
        
        if similaridades and similaridades[0]['score'] > 0.4:  # Threshold de confiança
            return categorias_disponiveis[similaridades[0]['corpus_id']]
        
        return None
    
    async def _filtrar_livros_candidatos(
        self, 
        categoria_filtro: Optional[str], 
        preco_max: Optional[float]
    ) -> pd.DataFrame:
        """Filtrar livros candidatos baseado nos critérios"""
        livros_candidatos = self.livros_df.copy()
        
        # Filtro por categoria
        if categoria_filtro:
            livros_candidatos = livros_candidatos[
                livros_candidatos["Categoria"].str.lower() == categoria_filtro.lower()
            ]
        
        # Filtro por preço
        if preco_max and 'Preco' in livros_candidatos.columns:
            livros_candidatos = livros_candidatos[
                livros_candidatos["Preco"] <= preco_max
            ]
        
        return livros_candidatos
    
    async def _busca_semantica(
        self, 
        consulta: str, 
        livros_candidatos: pd.DataFrame, 
        top_k: int
    ) -> List[Dict]:
        """Realizar busca semântica nos livros candidatos"""
        if livros_candidatos.empty:
            return []
        
        # Preparar embeddings dos candidatos
        indices_candidatos = livros_candidatos.index.tolist()
        embeddings_candidatos = self.embeddings_cache['livros_embeddings'][indices_candidatos]
        
        # Busca semântica
        consulta_embedding = self.modelo.encode(consulta, convert_to_tensor=True)
        resultados = util.semantic_search(
            consulta_embedding, 
            embeddings_candidatos, 
            top_k=top_k
        )[0]
        
        # Converter para formato padronizado
        resultados_formatados = []
        for r in resultados:
            idx_original = indices_candidatos[r['corpus_id']]
            livro = self.livros_df.loc[idx_original]
            
            resultado = {
                'produto_id': str(livro.get('ID_Produto', idx_original)),
                'titulo': livro['Titulo'],
                'categoria': livro['Categoria'],
                'descricao': livro['Descricao'],
                'preco': livro.get('Preco', None),
                'relevancia_score': float(r['score']),
                'indice_original': idx_original
            }
            resultados_formatados.append(resultado)
        
        return resultados_formatados
    
    async def _diversificar_e_rankear(
        self, 
        resultados: List[Dict], 
        top_k: int, 
        consulta: str
    ) -> List[Dict]:
        """Diversificar resultados e aplicar ranking final"""
        if not resultados:
            return []
        
        # Diversificação por categoria
        categorias_vistas = set()
        resultados_diversificados = []
        
        # Primeiro, pegar o melhor de cada categoria
        for resultado in resultados:
            categoria = resultado['categoria']
            if categoria not in categorias_vistas:
                resultados_diversificados.append(resultado)
                categorias_vistas.add(categoria)
                if len(resultados_diversificados) >= top_k:
                    break
        
        # Completar com os melhores restantes
        if len(resultados_diversificados) < top_k:
            for resultado in resultados:
                if resultado not in resultados_diversificados:
                    resultados_diversificados.append(resultado)
                    if len(resultados_diversificados) >= top_k:
                        break
        
        return resultados_diversificados[:top_k]
    
    async def _gerar_explicacoes(
        self, 
        recomendacoes: List[Dict], 
        consulta: str, 
        usuario_id: Optional[str]
    ) -> List[RecomendacaoResponse]:
        """Gerar explicações para as recomendações"""
        recomendacoes_final = []
        
        for rec in recomendacoes:
            # Calcular estatísticas do produto
            produto_id = rec['produto_id']
            avaliacoes = self.feedbacks_df[
                self.feedbacks_df['ID_Produto'] == produto_id
            ]['Avaliacao']
            
            avaliacao_media = float(avaliacoes.mean()) if not avaliacoes.empty else None
            numero_avaliacoes = len(avaliacoes)
            
            # Gerar explicação
            explicacao = await self._gerar_explicacao_individual(rec, consulta, usuario_id)
            
            # Gerar tags
            tags = self._gerar_tags(rec)
            
            recomendacao_response = RecomendacaoResponse(
                produto_id=rec['produto_id'],
                titulo=rec['titulo'],
                categoria=rec['categoria'],
                descricao=rec['descricao'][:200] + "..." if len(rec['descricao']) > 200 else rec['descricao'],
                preco=rec.get('preco'),
                avaliacao_media=avaliacao_media,
                numero_avaliacoes=numero_avaliacoes,
                relevancia_score=rec['relevancia_score'],
                explicacao=explicacao,
                tags=tags
            )
            
            recomendacoes_final.append(recomendacao_response)
        
        return recomendacoes_final
    
    async def _gerar_explicacao_individual(
        self, 
        recomendacao: Dict, 
        consulta: str, 
        usuario_id: Optional[str]
    ) -> str:
        """Gerar explicação individual para uma recomendação"""
        explicacoes = []
        
        # Explicação baseada na similaridade
        score = recomendacao['relevancia_score']
        if score > 0.8:
            explicacoes.append("Muito relevante para a sua consulta")
        elif score > 0.6:
            explicacoes.append("Relevante para a sua consulta")
        else:
            explicacoes.append("Relacionado com a sua consulta")
        
        # Explicação baseada na categoria
        categoria = recomendacao['categoria']
        if categoria in self.padroes_categoria:
            freq = self.padroes_categoria[categoria]
            if freq > 50:  # Threshold arbitrário
                explicacoes.append(f"Categoria popular ({categoria})")
        
        # Explicação baseada no histórico do usuário
        if usuario_id and usuario_id in self.perfis_usuarios:
            perfil = self.perfis_usuarios[usuario_id]
            if categoria in perfil.get('categorias_preferidas', []):
                explicacoes.append("Baseado nas suas preferências anteriores")
        
        return " • ".join(explicacoes) if explicacoes else "Recomendação baseada em similaridade"
    
    def _gerar_tags(self, recomendacao: Dict) -> List[str]:
        """Gerar tags para uma recomendação"""
        tags = []
        
        # Tag da categoria
        tags.append(recomendacao['categoria'].lower().replace(' ', '-'))
        
        # Tags baseadas na descrição
        descricao = recomendacao['descricao'].lower()
        palavras_chave = ['aventura', 'romance', 'mistério', 'história', 'fantasia', 'ciência']
        
        for palavra in palavras_chave:
            if palavra in descricao:
                tags.append(palavra)
        
        # Tag de preço
        preco = recomendacao.get('preco')
        if preco:
            if preco < 15:
                tags.append('económico')
            elif preco > 30:
                tags.append('premium')
        
        return tags[:5]  # Limitar a 5 tags
    
    async def _registrar_interacao(
        self, 
        usuario_id: Optional[str], 
        consulta: str, 
        recomendacoes: List[RecomendacaoResponse]
    ):
        """Registrar interação do usuário"""
        if not usuario_id:
            return
        
        interacao = {
            'timestamp': datetime.now(),
            'consulta': consulta,
            'recomendacoes': [r.produto_id for r in recomendacoes],
            'tipo': 'recomendacao'
        }
        
        self.historico_usuarios[usuario_id].append(interacao)
        
        # Atualizar perfil do usuário
        await self._atualizar_perfil_usuario(usuario_id, consulta, recomendacoes)
    
    async def _atualizar_perfil_usuario(
        self, 
        usuario_id: str, 
        consulta: str, 
        recomendacoes: List[RecomendacaoResponse]
    ):
        """Atualizar perfil do usuário com base na interação"""
        if usuario_id not in self.perfis_usuarios:
            self.perfis_usuarios[usuario_id] = {
                'categorias_preferidas': Counter(),
                'consultas_recentes': [],
                'produtos_vistos': set(),
                'primeira_interacao': datetime.now()
            }
        
        perfil = self.perfis_usuarios[usuario_id]
        
        # Atualizar categorias preferidas
        for rec in recomendacoes:
            perfil['categorias_preferidas'][rec.categoria] += 1
        
        # Atualizar consultas recentes
        perfil['consultas_recentes'].append(consulta)
        if len(perfil['consultas_recentes']) > 10:
            perfil['consultas_recentes'] = perfil['consultas_recentes'][-10:]
        
        # Atualizar produtos vistos
        for rec in recomendacoes:
            perfil['produtos_vistos'].add(rec.produto_id)
    
    async def processar_feedback(
        self,
        usuario_id: str,
        produto_id: str,
        tipo_feedback: TipoFeedback,
        avaliacao: Optional[int] = None,
        comentario: Optional[str] = None
    ) -> bool:
        """Processar feedback do usuário"""
        try:
            # Registrar feedback
            feedback = {
                'timestamp': datetime.now(),
                'usuario_id': usuario_id,
                'produto_id': produto_id,
                'tipo_feedback': tipo_feedback.value,
                'avaliacao': avaliacao,
                'comentario': comentario
            }
            
            # Adicionar ao histórico
            self.historico_usuarios[usuario_id].append({
                **feedback,
                'tipo': 'feedback'
            })
            
            # Processar no sistema de refinamento
            await self.refinamento.processar_feedback(feedback)
            
            # Atualizar estatísticas
            self.estatisticas['total_feedbacks'] += 1
            
            # Calcular taxa de aceitação
            feedbacks_positivos = sum(1 for hist in self.historico_usuarios[usuario_id] 
                                    if hist.get('tipo') == 'feedback' and 
                                    hist.get('tipo_feedback') in ['curtiu', 'comprou', 'salvou'])
            total_feedbacks_usuario = sum(1 for hist in self.historico_usuarios[usuario_id] 
                                        if hist.get('tipo') == 'feedback')
            
            if total_feedbacks_usuario > 0:
                taxa_usuario = feedbacks_positivos / total_feedbacks_usuario
                # Atualizar taxa global (média móvel simples)
                self.estatisticas['taxa_aceitacao'] = (
                    self.estatisticas['taxa_aceitacao'] * 0.9 + taxa_usuario * 0.1
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao processar feedback: {str(e)}")
            return False
    
    async def obter_historico_usuario(self, usuario_id: str, limit: int = 20) -> List[Dict]:
        """Obter histórico de um usuário"""
        if usuario_id not in self.historico_usuarios:
            return []
        
        historico = self.historico_usuarios[usuario_id]
        historico_ordenado = sorted(historico, key=lambda x: x['timestamp'], reverse=True)
        
        return historico_ordenado[:limit]
    
    async def obter_estatisticas(self) -> Dict[str, Any]:
        """Obter estatísticas do sistema"""
        tempo_medio = (
            sum(self.estatisticas['tempo_resposta']) / len(self.estatisticas['tempo_resposta'])
            if self.estatisticas['tempo_resposta'] else 0
        )
        
        return {
            'total_usuarios': len(self.historico_usuarios),
            'total_recomendacoes': self.estatisticas['total_recomendacoes'],
            'total_feedbacks': self.estatisticas['total_feedbacks'],
            'taxa_aceitacao': round(self.estatisticas['taxa_aceitacao'], 3),
            'categorias_populares': list(self.padroes_categoria.items())[:10],
            'modelo_stats': {
                'modelo_nome': 'paraphrase-multilingual-MiniLM-L12-v2',
                'total_livros': len(self.livros_df),
                'embeddings_cache': len(self.embeddings_cache)
            },
            'performance_metrics': {
                'tempo_resposta_medio_ms': round(tempo_medio, 2),
                'cache_hit_rate': 0.85,  # Placeholder
                'precisao_estimada': 0.78,  # Placeholder
                'diversidade_media': 0.65   # Placeholder
            }
        }
    
    def status_modelo(self) -> str:
        """Retornar status do modelo"""
        if self.modelo is None:
            return "não_inicializado"
        return "ok"