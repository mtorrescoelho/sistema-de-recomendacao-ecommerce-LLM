import requests
from bs4 import BeautifulSoup
import csv
import random
from datetime import datetime, timedelta
import os  
import time


# Diretório de saída para os ficheiros CSV
output_dir = 'dados'  # [ADICIONADO]
os.makedirs(output_dir, exist_ok=True)

# Função para guardar os livros em CSV (Separação de responsabilidades)
def guardar_csv_livros(livros, caminho=os.path.join(output_dir, 'livros.csv')): 
    with open(caminho, 'w', newline='', encoding='utf-8') as f:  
        writer = csv.DictWriter(f, fieldnames=['ID_Produto', 'Produto', 'Titulo', 'Categoria', 'Descricao', 'Preco', 'Avaliacao', 'Disponibilidade'])
        writer.writeheader()
        writer.writerows(livros)
    print(f"Dados dos livros guardados em '{caminho}'")


# Scraping de livros do site "Books to Scrape"
def scrapear_livros():

    # URL do site dos livros (mantendo o formato original)
    base_url = 'https://books.toscrape.com/catalogue/page-{}.html'
    books = []
    #criar um contador para o id_produto
    id_produto = 1


    # Função para gerar um ID de utilizador (simulação)
    def gerar_id_utilizador():
        return random.randint(1, 100)

    # Função para converter texto da classe em número de estrelas
    def estrelas_para_numero(texto):
        estrelas_dict = {
            'One': 1,
            'Two': 2,
            'Three': 3,
            'Four': 4,
            'Five': 5
        }
        return estrelas_dict.get(texto, 0)

    # Percorrer páginas do site (50 no total)
    for page in range(1, 51):
        print(f'A processar página {page}...')
        try:  # [ADICIONADO: retry com timeout]
            response = requests.get(base_url.format(page), timeout=10)
            response.encoding = 'utf-8'
            response.raise_for_status()
        except requests.RequestException as e:
            print(f'Erro na página {page}: {e}')
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        livros = soup.find_all('article', class_='product_pod')

        for livro in livros:
            titulo = livro.h3.a['title']
            preco_texto = livro.find('p', class_='price_color').text.strip().replace('Â', '').replace('£', '')
            try:
                preco = float(preco_texto)
            except ValueError:
                print(f"Erro ao converter preço: '{preco_texto}' — livro ignorado.")
                continue
            preco = float(preco_texto.replace('£', '')) #  limpeza e conversão para float
            disponibilidade = livro.find('p', class_='instock availability').text.strip()
            estrelas_classe = livro.find('p', class_='star-rating')['class'][1]
            estrelas = estrelas_para_numero(estrelas_classe)

            # Ir à página interna do produto para obter a categoria e descrição
            link_parcial = livro.h3.a['href']
            link_completo = 'https://books.toscrape.com/catalogue/' + link_parcial

        
            #detalhes = requests.get(link_completo)
            try: #timeout na página de detalhes
                detalhes = requests.get(link_completo, timeout=10)
                detalhes.encoding = 'utf-8'
                detalhes.raise_for_status()
            except requests.RequestException as e:
                print(f'Erro ao aceder ao link {link_completo}: {e}')
                continue


            soup_detalhes = BeautifulSoup(detalhes.text, 'html.parser')

            #categoria
            categoria = soup_detalhes.find('ul', class_='breadcrumb').find_all('a')[2].text

            # Descrição
            descricao_tag = soup_detalhes.find('meta', attrs={'name': 'description'})
            descricao = descricao_tag['content'].strip() if descricao_tag else 'Sem descrição'
            
            books.append({
                'ID_Produto': id_produto,
                'Produto': 'Livro',
                'Titulo': titulo,
                'Categoria': categoria,
                'Descricao': descricao, 
                'Preco': preco,
                'Avaliacao': estrelas,
                'Disponibilidade': disponibilidade
            })
            id_produto += 1
            
            time.sleep(1)

    # Guardar para CSV
    guardar_csv_livros(books) 
    print("Scraping concluído. Dados guardados em 'livros.csv'")
    return books
  
# ==============Gerar FEEDBACKS aleatórios===============================================
def gerar_feedbacks(livros):
    feedbacks = []

    # Feedbacks de exemplo
    feedback_bom = [
        "Produto excelente, recomendo!",
        "Muito satisfeito com a compra.",
        "Qualidade incrível, voltarei a comprar.",
        "Atendeu todas as minhas expectativas.",
        "Ótimo custo-benefício, vale a pena.",
        "Produto de alta qualidade, superou minhas expectativas.",
        "Recomendo a todos, muito bom!",
        "Produto chegou rápido e em perfeito estado.",
        "Estou muito feliz com a compra, excelente produto.",
        "Produto de ótima qualidade, muito satisfeito.",
        "Produto chegou antes do prazo, muito bom!",
        "Produto de qualidade, atendeu minhas expectativas.",
        "Produto muito bom, recomendo a todos.",
        "Produto excelente, muito satisfeito com a compra.",
        "Produto de alta qualidade, recomendo!",
    ]
    feedback_medio = [
        "Produto bom, mas poderia ser melhor.",
        "Atendeu parcialmente minhas expectativas.",
        "Não é o que eu esperava, mas ainda assim é aceitável.",
        "A qualidade é média, mas o preço é justo.",
        "Produto razoável, mas não voltaria a comprar.",
        "O produto é bom, mas o atendimento poderia ser melhor.",
        "Produto bom, mas o prazo de entrega foi longo.",
        "Produto atende ao que promete, mas não é excepcional.",
        "A qualidade é boa, mas o preço poderia ser menor.",
        "Produto razoável, mas não é o melhor do mercado.",
        "Produto bom, mas não é o que eu esperava.",
        "Cumpre o que promete, mas nada além disso."
    ]
    feedback_mau = [
        "Produto muito ruim, não recomendo.",
        "Totalmente insatisfeito com a compra.",
        "Não vale o preço que paguei.",
        "A qualidade é péssima, não compensa.",
        "Produto chegou danificado, não gostei.",
        "Não funcionou como esperado, muito decepcionado.",
        "Produto não corresponde à descrição.",
        "A entrega atrasou e o produto chegou quebrado.",
        "Não gostei do produto, não recomendo.",
        "Produto de baixa qualidade, não vale a pena.",
        "Não funcionou como esperado, decepcionante."
    ]

    # Gerar feedbacks aleatórios para cada livro
    for livro in livros:
        num_feedbacks = random.choices([0, 1, 3, 5, 10, 15], weights=[10, 5, 25, 20, 15, 5])[0]
        
        for _ in range(num_feedbacks):
            user_id = random.randint(1, 100)
            produto_id = livro['ID_Produto']

            # Escolha de sentimento com base na avaliação do livro
            avaliacao_livro = livro['Avaliacao']
            if avaliacao_livro >= 4:
                # Livro com avaliação alta: mais chance de feedback bom
                sentimento = random.choices(['bom', 'medio', 'mau'], weights=[70, 20, 10])[0]
            elif avaliacao_livro == 3:
                # Livro com avaliação média: distribuição equilibrada
                sentimento = random.choices(['bom', 'medio', 'mau'], weights=[30, 50, 20])[0]
            else:
                # Livro com avaliação baixa: mais chance de feedback mau
                sentimento = random.choices(['bom', 'medio', 'mau'], weights=[10, 30, 60])[0]

            if sentimento == 'bom':
                texto = random.choice(feedback_bom)
                estrelas = random.randint(4, 5)
            elif sentimento == 'medio':
                texto = random.choice(feedback_medio)
                estrelas = 3
            else:
                texto = random.choice(feedback_mau)
                estrelas = random.randint(1, 2)

            data_feedback = datetime.now() - timedelta(days=random.randint(0, 365))
            data_compra = data_feedback - timedelta(days=random.randint(1, 30))

            feedbacks.append({
                'ID_Utilizador': user_id,
                'ID_Produto': produto_id,
                'Feedback_Texto': texto,
                'Avaliacao': estrelas,
                'Data_Feedback': data_feedback.strftime('%Y-%m-%d'),
                'Data_Compra': data_compra.strftime('%Y-%m-%d'),
            })

            
    caminho_feedback = os.path.join(output_dir, 'feedbacks.csv')
    # Guardar feedbacks em CSV
    with open(caminho_feedback, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['ID_Produto', 'ID_Utilizador' , 'Data_Compra' , 'Avaliacao' , 'Feedback_Texto', 'Data_Feedback'])
        writer.writeheader()
        writer.writerows(feedbacks)

    print(f"Feedbacks gerados e guardados em '{caminho_feedback}'")

if __name__ == "__main__":
    # Executar o scraping e gerar feedbacks
    livros = scrapear_livros()
    gerar_feedbacks(livros)
