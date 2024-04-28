# Iris Classification Project

## Descrição
Projeto para análise e classificação do dataset Iris utilizando Python e machine learning. Inclui uma API FastAPI para realizar previsões.

## Estrutura de Diretórios
- `src/`: Código-fonte para o projeto.
- `api/`: API FastAPI que serve o modelo.
- `model/`: Scripts para treinamento e o modelo treinado.
- `Dockerfile`: Para criar um ambiente Docker reproduzível.
- `requirements.txt`: Lista de dependências do projeto.

## Como Usar

### Usando python no terminal
1. Clone o repositório.
2. Instale as dependências `pip install -r requirements.txt`
3. Execute o código da API `python src/main.py`.
4. Acesse a API em `http://localhost:8000`.

### Usando docker
1. Clone o repositório.
2. Construa o container Docker com `docker build -t iris-project .`.
3. Execute o container com `docker run -p 8000:8000 iris-project`.
4. Acesse a API em `http://localhost:8000`.

### Exemplo de Requisição
Usando `curl` para fazer uma requisição POST para a API:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'