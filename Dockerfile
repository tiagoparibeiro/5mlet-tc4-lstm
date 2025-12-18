# ===============================================
# STAGE 1: Configuração do Ambiente
# ===============================================

# 1. Imagem Base: Usamos uma imagem Python leve (slim)
#    Recomenda-se usar 'slim' para reduzir o tamanho da imagem final.
FROM python:3.11-slim

# 2. Definir variáveis de ambiente para desativar o buffer do Python,
#    o que é bom para logs em tempo real em contêineres.
ENV PYTHONUNBUFFERED 1

# 3. Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# ===============================================
# STAGE 2: Instalação de Dependências
# ===============================================

# 4. Copiar apenas o arquivo de requisitos primeiro.
#    Isso otimiza o cache do Docker, pois raramente alteramos os requisitos.
COPY requirements.txt .

# 5. Instalar as dependências.
#    '--no-cache-dir' evita que o pip armazene pacotes, economizando espaço.
#    O TensorFlow é grande, então este passo pode levar alguns minutos.
RUN pip install --no-cache-dir -r requirements.txt

# ===============================================
# STAGE 3: Copiar Código e Ativos
# ===============================================

# 6. Copiar o diretório da API e seus conteúdos.
COPY api /app/api

# 7. Criar a pasta onde o modelo será baixado (vazia por enquanto)
RUN mkdir -p /app/model

# 8. Copiar outros arquivos essenciais (como o README, se necessário)
# COPY README.md /app/README.md

# ===============================================
# STAGE 4: Comando de Execução
# ===============================================

# 9. Expor a porta que o Uvicorn irá escutar
EXPOSE 8000

# 10. Comando principal para iniciar a aplicação
#     Usamos Gunicorn como um gerenciador de processos em produção,
#     junto com Uvicorn workers, para maior robustez e paralelismo.
#     O comando abaixo inicializa 4 workers (ajuste conforme o número de CPUs)
CMD ["gunicorn", "api.main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# ALTERNATIVA SIMPLES (para ambientes com poucos recursos ou testes):
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]