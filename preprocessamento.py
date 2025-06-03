import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from ftfy import fix_text
import os

# ========== 1. Limpeza e normalização de texto ==========

def normalizar_texto(texto):
    if pd.isnull(texto):
        return ""

    texto = fix_text(texto)
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    return texto

def limpar_descricao(desc):
    # Tira conteúdo duplicado
    metades = desc.strip().split()
    metade = " ".join(metades[:len(metades)//2])

    # Tira palavras repetidas 3x ou mais
    metade = re.sub(r'\b(\w+)( \1\b)+', r'\1', metade)

    # Corrige espaços e coloca ponto final se faltar
    metade = re.sub(r'\s+', ' ', metade).strip()
    if not metade.endswith('.'):
        metade += '.'
    
    return metade.capitalize()


def limpar_dados_livros(df):
    colunas_necessarias = ['Titulo', 'Categoria', 'Descricao', 'Preco']
    for col in colunas_necessarias:
        if col not in df.columns:
            raise ValueError(f"A coluna '{col}' é obrigatória na tabela de livros.")

    df = df.dropna(subset=colunas_necessarias)
    df['Categoria'] = df['Categoria'].astype(str).str.lower().str.strip()

    categorias_invalidas = ['default', 'n/a', 'sem categoria', 'desconhecido', 'add a comment', 'none', '']
    df = df[~df['Categoria'].isin(categorias_invalidas)]

    df['Preco'] = df['Preco'].astype(str).str.replace(',', '.', regex=False).astype(float)
    df['Avaliacao'] = pd.to_numeric(df['Avaliacao'], errors='coerce').fillna(0).astype(int)

    df['Descricao'] = df['Descricao'].apply(limpar_descricao)


    return df

def tokenizar_descricoes(df, coluna="Descricao", max_features=500):
    if coluna not in df.columns:
        raise ValueError(f"A coluna '{coluna}' é obrigatória para o TF-IDF.")
    df[coluna] = df[coluna].astype(str).apply(normalizar_texto)
    tfidf = TfidfVectorizer(max_features=max_features)
    matriz_tfidf = tfidf.fit_transform(df[coluna])
    return matriz_tfidf, tfidf

# ========== 2. Variáveis derivadas ==========

def calcular_sazonalidade(df, data_col='Data_Compra'):
    if data_col not in df.columns:
        raise ValueError(f"A coluna '{data_col}' é obrigatória para calcular a sazonalidade.")
    df[data_col] = pd.to_datetime(df[data_col], errors='coerce')
    if df[data_col].isnull().any():
        raise ValueError(f"Existem valores inválidos na coluna '{data_col}'.")

    df['ano_mes'] = df[data_col].dt.to_period('M')
    sazonalidade = df.groupby('ano_mes').size().reset_index(name='compras_no_mes')
    return sazonalidade

def calcular_fidelizacao(df, cliente_col='ID_Utilizador'):
    if cliente_col not in df.columns:
        raise ValueError(f"A coluna '{cliente_col}' é obrigatória para calcular fidelização.")
    compras_por_cliente = df.groupby(cliente_col).size()
    clientes_fieis = compras_por_cliente[compras_por_cliente > 1].count()
    taxa_fidelizacao = clientes_fieis / df[cliente_col].nunique()
    return taxa_fidelizacao

def calcular_valor_medio_por_compra(df, valor_col='Preco', cliente_col='ID_Utilizador'):
    if cliente_col not in df.columns or valor_col not in df.columns:
        raise ValueError(f"As colunas '{cliente_col}' e '{valor_col}' são obrigatórias para calcular valor médio.")
    return df.groupby(cliente_col)[valor_col].mean().reset_index(name='valor_medio_por_compra')

def calcular_intervalo_medio_entre_compras(df, cliente_col='ID_Utilizador', data_col='Data_Compra'):
    if cliente_col not in df.columns or data_col not in df.columns:
        raise ValueError(f"As colunas '{cliente_col}' e '{data_col}' são obrigatórias para calcular intervalos.")
    df[data_col] = pd.to_datetime(df[data_col], errors='coerce')
    if df[data_col].isnull().any():
        raise ValueError(f"Existem datas inválidas na coluna '{data_col}'.")

    df = df.sort_values(by=[cliente_col, data_col])
    intervalos = df.groupby(cliente_col)[data_col].diff().dt.days
    intervalo_medio = intervalos.groupby(df[cliente_col]).mean().reset_index(name='intervalo_medio_dias')
    return intervalo_medio

# ========== 3. Função principal de pré-processamento ==========

def preprocessar_dados(df):
    # Normalizar texto antes do TF-IDF
    if 'Descricao' not in df.columns:
        raise ValueError("Coluna 'Descricao' em falta para normalização e TF-IDF.")
    matriz_tfidf, tfidf_model = tokenizar_descricoes(df)

    sazonalidade = calcular_sazonalidade(df)
    fidelizacao = calcular_fidelizacao(df)
    valor_medio = calcular_valor_medio_por_compra(df)
    intervalo_medio = calcular_intervalo_medio_entre_compras(df)

    return {
        "matriz_tfidf": matriz_tfidf,
        "tfidf_model": tfidf_model,
        "sazonalidade": sazonalidade,
        "taxa_fidelizacao": fidelizacao,
        "valor_medio": valor_medio,
        "intervalo_medio": intervalo_medio
    }

# ========== 4. Execução direta ==========

if __name__ == "__main__":
    livros = pd.read_csv("dados/livros.csv")
    feedbacks = pd.read_csv("dados/feedbacks.csv")

    livros_limpos = limpar_dados_livros(livros)
    livros_limpos.to_csv("dados/livros_limpos.csv", index=False)

    df = pd.merge(feedbacks, livros_limpos, on="ID_Produto", how="left")
    resultados = preprocessar_dados(df)

    os.makedirs("outputs_p", exist_ok=True)
    resultados["sazonalidade"].to_csv("outputs_p/sazonalidade.csv", index=False)
    resultados["valor_medio"].to_csv("outputs_p/valor_medio_por_compra.csv", index=False)
    resultados["intervalo_medio"].to_csv("outputs_p/intervalo_medio_entre_compras.csv", index=False)

    print("Pré-processamento concluído. Ficheiros guardados:")
    print(" - livros_limpos.csv")
    print(" - feedbacks.csv")
    print(" - sazonalidade.csv")
    print(" - valor_medio_por_compra.csv")
    print(" - intervalo_medio_entre_compras.csv")

    print("\nTF-IDF (primeiras 5 linhas):")
    print(resultados["matriz_tfidf"][:5])

    print("\nValor médio por cliente (primeiras 5 linhas):")
    print(resultados["valor_medio"].head())

    print("\nTexto normalizado (primeiras 5 descrições):")
    print(df["Descricao"].head())
