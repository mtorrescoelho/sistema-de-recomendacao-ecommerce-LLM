import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
from pathlib import Path
def carregar_dados_modelo():
    livros_df = pd.read_csv("dados/livros_limpos.csv")
    feedbacks_df = pd.read_csv('dados/feedbacks.csv')
    modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return livros_df, feedbacks_df, modelo

def extrair_categoria_frase(frase, categorias_disponiveis):
    frase_lower = frase.lower()
    for categoria in categorias_disponiveis:
        if categoria.lower() in frase_lower:
            return categoria
    return None

def recomendar_livros_compreensivo(pergunta, livros_df, feedbacks_df, modelo, top_k=5):
    categorias_disponiveis = livros_df["Categoria"].unique()
    categoria_detectada = extrair_categoria_frase(pergunta, categorias_disponiveis)

    # Se foi detectada categoria, filtra os livros dessa categoria
    if categoria_detectada:
        livros_candidatos = livros_df[livros_df["Categoria"].str.lower() == categoria_detectada.lower()]
    else:
        livros_candidatos = livros_df

    if livros_candidatos.empty:
        return [f"Nenhum livro encontrado na categoria '{categoria_detectada}'."], categoria_detectada

    # Construir os textos descritivos dos livros
    textos_livros = (
        livros_candidatos["Titulo"] + " - Categoria: " + livros_candidatos["Categoria"] + " - " + livros_candidatos["Descricao"]
    ).tolist()

    embeddings_livros = modelo.encode(textos_livros, convert_to_tensor=True)
    embedding_pergunta = modelo.encode(pergunta, convert_to_tensor=True)

    resultados = util.semantic_search(embedding_pergunta, embeddings_livros, top_k=top_k)[0]

    recomendacoes = []
    for r in resultados:
        idx_df = livros_candidatos.index[r["corpus_id"]]
        titulo = livros_candidatos.loc[idx_df, "Titulo"]
        categoria = livros_candidatos.loc[idx_df, "Categoria"]
        descricao = livros_candidatos.loc[idx_df, "Descricao"]

        # Obter média de avaliação (opcional)
        pid = livros_candidatos.loc[idx_df, "ID_Produto"]
        avaliacoes = feedbacks_df[feedbacks_df["ID_Produto"] == pid]["Avaliacao"]
        media_avaliacao = avaliacoes.mean() if not avaliacoes.empty else "Sem avaliações"

        texto = (
            f"📘 {titulo}\n"
            f"Categoria: {categoria}\n"
            f"Descrição: {descricao[:200]}...\n"
            f"Avaliação média: {media_avaliacao}\n"
        )
        recomendacoes.append(texto)

    return recomendacoes, categoria_detectada

def responder_com_chatbot(pergunta, livros_df, feedbacks_df, modelo):
    recomendacoes, categoria = recomendar_livros_compreensivo(pergunta, livros_df, feedbacks_df, modelo)

    if not recomendacoes:
        return "Não encontrei livros relevantes. Podes tentar reformular?"

    resposta = "Aqui estão algumas sugestões que podem interessar-te:\n\n"
    for r in recomendacoes:
        resposta += r + "\n---\n"

    resposta += "\nSe quiseres mais sugestões ou procuras outro tema, é só dizeres!"
    return resposta

def main():
    livros_df, feedbacks_df, modelo = carregar_dados_modelo()
    while True:
        pergunta = input("🤖> O que procuras? (ou escreve 'sair')\n> ")
        if pergunta.lower() in ["sair", "exit"]:
            break
        print("\n" + responder_com_chatbot(pergunta, livros_df, feedbacks_df, modelo))

if __name__ == "__main__":
    main()