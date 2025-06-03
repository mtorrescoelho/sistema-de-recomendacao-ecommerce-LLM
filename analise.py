import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
livros = pd.read_csv('dados/livros_limpos.csv')
feedbacks = pd.read_csv('dados/feedbacks.csv')

# Converter datas
feedbacks['Data_Compra'] = pd.to_datetime(feedbacks['Data_Compra'])
feedbacks['Data_Feedback'] = pd.to_datetime(feedbacks['Data_Feedback'])

# Juntar os datasets
df = pd.merge(feedbacks, livros, on='ID_Produto', how='left')

# Criar pasta para guardar os outputs (opcional)
import os
os.makedirs("outputs_a", exist_ok=True)

# ========== ANÁLISE 1: Produtos mais vendidos ==========
produtos_mais_vendidos = df['Titulo'].value_counts().head(10)
print("📚 Top 10 produtos mais vendidos (por nº de feedbacks):")
print(produtos_mais_vendidos)

plt.figure(figsize=(12,6))
sns.barplot(x=produtos_mais_vendidos.values, y=produtos_mais_vendidos.index, palette='viridis')
plt.title('Top 10 Produtos Mais Vendidos')
plt.xlabel('Nº de Feedbacks (simulação de vendas)')
plt.ylabel('Título do Produto')
plt.tight_layout()
plt.savefig('outputs_a/top_produtos.png')
plt.close()

# Guardar CSV
produtos_mais_vendidos.to_csv('outputs_a/top_produtos.csv', header=True)

# ========== ANÁLISE 2: Categorias mais populares ==========
categorias_populares = df['Categoria'].value_counts().head(10)
print("\n📚 Categorias mais populares:")
print(categorias_populares)

plt.figure(figsize=(10,6))
sns.barplot(x=categorias_populares.values, y=categorias_populares.index, palette='magma')
plt.title('Top 10 Categorias Mais Populares')
plt.xlabel('Nº de Feedbacks')
plt.ylabel('Categoria')
plt.tight_layout()
plt.savefig('outputs_a/categorias_populares.png')
plt.close()

categorias_populares.to_csv('outputs_a/categorias_populares.csv', header=True)

# ========== ANÁLISE 3: Nº de compras ao longo do tempo ==========
compras_por_mes = df.groupby(df['Data_Compra'].dt.to_period('M')).size()

plt.figure(figsize=(12,6))
compras_por_mes.plot(kind='line', marker='o')
plt.title('Nº de Compras ao Longo do Tempo')
plt.xlabel('Mês')
plt.ylabel('Nº de Compras')
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs_a/compras_por_mes.png')
plt.close()

compras_por_mes = compras_por_mes.sort_index()
compras_por_mes.to_csv('outputs_a/compras_por_mes.csv', header=True)



# ========== ANÁLISE 4: Distribuição dos preços ==========
livros['Preco'] = livros['Preco'].apply(lambda x: str(x).replace('£', '') if pd.notnull(x) else x)
livros['Preco'] = pd.to_numeric(livros['Preco'], errors='coerce')


plt.figure(figsize=(10,6))
sns.histplot(livros['Preco'], bins=20, kde=True, color='skyblue')
plt.title('Distribuição dos Preços dos Produtos')
plt.xlabel('Preço (£)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig('outputs_a/distribuicao_precos.png')
plt.close()

livros['Preco'].describe().to_csv('outputs_a/estatisticas_precos.csv', header=True)