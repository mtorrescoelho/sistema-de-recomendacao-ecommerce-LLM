class TextProcessor:
    def criar_texto_descritivo(self, livro):
        # Junta título, categoria e descrição
        return f"{livro['Titulo']} - Categoria: {livro['Categoria']} - {livro['Descricao']}"

    def processar_consulta(self, consulta):
        # Apenas retorna a consulta (pode melhorar depois)
        return consulta