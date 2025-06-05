from datetime import datetime


class SistemaRefinamento:
    def __init__(self):
        self.livros_limpos_df = None
        self.feedbacks_df = None
        self.categorias_populares = None
        self.compras_por_mes = None
        self.estatisticas_precos = None
        self.top_produtos = None
        self.sazonalidade = None
        self.valor_medio_por_compra = None

    async def inicializar(self, dados):
        # Carregar os dados calculados aqui 
        
        self.livros_df = dados.get('livros_limpos')
        self.feedbacks_df = dados.get('feedbacks')
        self.categorias_populares = dados.get('categorias_populares')
        self.compras_por_mes = dados.get('compras_por_mes')
        self.estatisticas_precos = dados.get('estatisticas_precos')
        self.top_produtos = dados.get('top_produtos')
        self.sazonalidade = dados.get('sazonalidade')
        self.valor_medio_por_compra = dados.get('valor_medio_por_compra')

    async def refinar_resultados(self, resultados, padroes_categoria=None, padroes_sazonalidade=None):
        # Exemplo: priorizar livros de categorias populares
        if self.categorias_populares is not None:
            resultados = sorted(
                resultados,
                key=lambda x: self.categorias_populares.get(x['Categoria'], 0),
                reverse=True
            )

        # Exemplo: dar preferência a livros em alta na sazonalidade atual
        if self.sazonalidade is not None:
            mes_atual = datetime.now().month
            resultados = sorted(
                resultados,
                key=lambda x: self.sazonalidade.get((x['Categoria'], mes_atual), 0),
                reverse=True
            )

        # Exemplo: destacar top produtos
        if self.top_produtos is not None:
            top_ids = set(self.top_produtos['ID_Produto'])
            resultados = sorted(
                resultados,
                key=lambda x: x['ID_Produto'] in top_ids,
                reverse=True
            )

        # Exemplo: filtrar por faixa de preço média do usuário (se disponível)
        # if self.valor_medio_por_compra is not None:
        #     resultados = [r for r in resultados if r['Preco'] <= self.valor_medio_por_compra + 5]

        return resultados