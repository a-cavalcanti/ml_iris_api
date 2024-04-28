FROM python:3.8-slim

# Definir o diretório de trabalho no container
WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar a pasta da API e do modelo
COPY api/ ./api/
COPY model/ ./model/

# Porta que a API usará
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]