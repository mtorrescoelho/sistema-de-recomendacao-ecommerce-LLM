import pandas as pd
from sentence_transformers import SentenceTransformer, util

def carregar_dados_modelo():
    df = pd.read_csv("dados/livros_limpos.csv")
    modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return df, modelo

def mapear_categoria(pergunta):
    texto = pergunta.lower()
    if any(p in texto for p in ["criança", "infantil", "miúdo", "miúda", "filho", "filha"]):
        return ["childrens"]
    elif "romance" in texto:
        return ["romance"]
    elif "ficção histórica" in texto or "histórica" in texto:
        return ["historical fiction"]
    elif any(p in texto for p in ["jovem", "young", "adolescente"]):
        return ["young adult"]
    elif "ficção" in texto:
        return ["fiction"]
    elif "fantasia" in texto:
        return ["fantasy"]
    elif "mistério" in texto:
        return ["mystery"]
    elif "comida" in texto or "culinária" in texto:
        return ["food and drink"]
    elif any(p in texto for p in ["não ficção", "nonfiction", "biografia"]):
        return ["nonfiction"]
    elif "drama" in texto:
        return ["fiction", "young adult", "historical fiction", "romance"]
    return []

def recomendar_livros(pergunta, df, modelo, top_k=10):
    if not pergunta.strip():
        return ["Por favor, indica o que procuras para te poder recomendar livros."]

    categorias_filtros = mapear_categoria(pergunta)

    if categorias_filtros:
        df_filtrado = df[df["Categoria"].str.lower().isin([c.lower() for c in categorias_filtros])]
    else:
        df_filtrado = df

    if df_filtrado.empty:
        return ["Não encontrei livros que correspondam à tua descrição. Tenta reformular!"]

    textos_livros = (df_filtrado["Titulo"] + " - Categoria: " + df_filtrado["Categoria"] + " - " + df_filtrado["Descricao"]).tolist()
    embeddings_filtrados = modelo.encode(textos_livros, convert_to_tensor=True)
    embedding_pergunta = modelo.encode(pergunta, convert_to_tensor=True)

    resultados = util.semantic_search(embedding_pergunta, embeddings_filtrados, top_k=min(top_k, len(textos_livros)))[0]

    recomendacoes = []
    for r in resultados:
        idx_df = df_filtrado.index[r["corpus_id"]]
        categoria = df_filtrado.loc[idx_df, "Categoria"]
        titulo = df_filtrado.loc[idx_df, "Titulo"]
        descricao = df_filtrado.loc[idx_df, "Descricao"]
        score = r["score"]
        recomendacoes.append(f"{titulo} ({categoria}) - {descricao} [score: {score:.3f}]")

    return recomendacoes

def responder_com_chatbot(pergunta, df, modelo):
    recomendados = recomendar_livros(pergunta, df, modelo)
    if len(recomendados) == 1 and "Não encontrei" in recomendados[0]:
        return "🤖 " + recomendados[0]
    resposta = "Claro! Com base no que disseste, recomendo os seguintes livros:\n"
    for livro in recomendados:
        resposta += f"📘 {livro}\n"
    resposta += "\nSe quiseres mais sugestões ou tens outro tema em mente, diz-me!"
    return resposta

def main():
    df, modelo = carregar_dados_modelo()
    while True:
        pergunta = input("🧠 O que procuras? (ou escreve 'sair')\n> ")
        if pergunta.lower() in ["sair", "exit"]:
            break
        print("\n" + responder_com_chatbot(pergunta, df, modelo))

if __name__ == "__main__":
    main()
