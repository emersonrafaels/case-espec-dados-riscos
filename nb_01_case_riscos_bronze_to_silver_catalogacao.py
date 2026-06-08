import sys
import re
import time
import logging
from io import BytesIO

import boto3
import botocore.exceptions
import pandas as pd
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import current_timestamp, lit


# ============================================================
# Configuração de Logging
# ============================================================

# Configura o logger raiz com nível INFO e formato estruturado.
# No AWS Glue, as mensagens são encaminhadas automaticamente ao CloudWatch Logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuração do Job
# ============================================================

# Lê parâmetros injetados pelo Glue no momento da execução do job
args = getResolvedOptions(sys.argv, ["JOB_NAME"])

# --- Identificação do bucket e database ---
BUCKET = "workspace-db-case-espec-dados-riscos"
DATABASE_NAME = "workspace_db_case_espec_dados_riscos"

# --- Prefixos das camadas no S3 (arquitetura medalhão) ---
BRONZE_PREFIX = "bronze"
SILVER_PREFIX = "silver"

# --- Limiar de nulos (0.0 a 1.0): colunas acima desse percentual são
#     reportadas no log de qualidade. Não bloqueia o processamento. ---
NULL_THRESHOLD = 0.5

# --- Pastas da camada Bronze a serem processadas neste job ---
SOURCE_FOLDERS = [
    "cmdb_ci_spkg_sot",
    "cmdb_software_instance_sot",
    "resultado_query3",
    "servidores_siglas"
]


# ============================================================
# Inicialização do Glue Job
# ============================================================

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Clientes boto3 instanciados uma única vez e reutilizados durante todo o job
s3 = boto3.client("s3")
glue = boto3.client("glue")


# ============================================================
# Funções auxiliares – Normalização de nomes
# ============================================================

def normalize_column_name(column_name: str) -> str:
    """
    Padroniza nomes de colunas para formato compatível com Athena/Glue:
    sem espaços, sem caracteres especiais, tudo em minúsculas.

    Exemplos:
        'Sigla SS'          -> 'sigla_ss'
        'Criticidade Final' -> 'criticidade_final'
        '  '                -> 'coluna_sem_nome'
    """
    # Garante que o valor é string e remove espaços nas bordas
    column_name = str(column_name).strip().lower()

    # Substitui qualquer sequência de caracteres não alfanuméricos (exceto _) por _
    column_name = re.sub(r"[^\w]+", "_", column_name)

    # Colapsa múltiplos underscores consecutivos em um único
    column_name = re.sub(r"_+", "_", column_name)

    # Remove underscores remanescentes nas bordas
    column_name = column_name.strip("_")

    # Garante que o nome nunca fique vazio após todas as substituições
    if not column_name:
        column_name = "coluna_sem_nome"

    return column_name


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica normalize_column_name a todas as colunas de um DataFrame Pandas.
    Resolve colisões de nomes duplicados adicionando sufixo numérico (_1, _2, ...).
    """
    normalized_columns = []
    # Dicionário para rastrear quantas vezes cada nome normalizado já apareceu
    seen = {}

    for column in df.columns:
        normalized = normalize_column_name(column)

        if normalized in seen:
            # Nome duplicado: incrementa contador e adiciona sufixo
            seen[normalized] += 1
            normalized = f"{normalized}_{seen[normalized]}"
        else:
            # Primeira ocorrência: registra no dicionário
            seen[normalized] = 0

        normalized_columns.append(normalized)

    df.columns = normalized_columns
    return df


# ============================================================
# Funções auxiliares – Leitura de dados do S3
# ============================================================

def list_files_from_s3_folder(bucket: str, prefix: str) -> list:
    """
    Lista todos os arquivos (objetos não-pasta) dentro de um prefixo S3.
    Usa paginação para suportar prefixos com mais de 1.000 objetos.
    """
    files = []
    # O paginador abstrai múltiplas chamadas à API quando há mais de 1.000 objetos
    paginator = s3.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Objetos cujo key termina em "/" são diretórios virtuais — ignorar
                if not key.endswith("/"):
                    files.append(key)
    except botocore.exceptions.ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        logger.error(
            "Falha ao listar objetos em s3://%s/%s — código boto3: %s",
            bucket, prefix, error_code
        )
        raise

    return files


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Lê bytes de um CSV em um DataFrame Pandas.
    Tenta UTF-8 (com suporte a BOM) primeiro; em caso de falha de encoding,
    faz fallback para Latin-1 — frequente em arquivos com caracteres brasileiros.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            # sep=None + engine='python' deixa o Pandas detectar o delimitador
            # automaticamente (vírgula, ponto-e-vírgula, tabulação, etc.)
            return pd.read_csv(
                BytesIO(file_bytes), sep=None, engine="python", encoding=encoding
            )
        except UnicodeDecodeError:
            logger.warning("Encoding '%s' falhou na leitura do CSV; tentando próximo...", encoding)

    raise ValueError("Não foi possível decodificar o CSV com nenhum encoding suportado.")


def _read_excel(file_bytes: bytes) -> pd.DataFrame:
    """
    Lê bytes de um arquivo XLSX em um DataFrame Pandas.
    Usa openpyxl, que suporta o formato .xlsx moderno (Office Open XML).
    """
    return pd.read_excel(BytesIO(file_bytes), engine="openpyxl")


def read_s3_file_to_pandas(bucket: str, key: str) -> pd.DataFrame:
    """
    Obtém um objeto do S3 e delega a leitura ao helper adequado
    conforme a extensão do arquivo (.csv ou .xlsx).
    Lança ValueError para extensões não suportadas.
    """
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        logger.error("Erro ao obter s3://%s/%s — %s", bucket, key, error_code)
        raise

    file_bytes = response["Body"].read()

    if key.lower().endswith(".csv"):
        return _read_csv(file_bytes)

    if key.lower().endswith(".xlsx"):
        return _read_excel(file_bytes)

    raise ValueError(f"Formato de arquivo não suportado: {key}")


# ============================================================
# Funções auxiliares – Qualidade de Dados
# ============================================================

def validate_dataframe(df: pd.DataFrame, source_key: str) -> bool:
    """
    Realiza validações básicas de qualidade no DataFrame lido da Bronze.

    - Verifica se o DataFrame está vazio (critério de rejeição).
    - Loga row count e column count.
    - Emite WARNING para colunas cujo percentual de nulos excede NULL_THRESHOLD.

    Retorna True se o DataFrame é válido para processamento, False caso contrário.
    """
    if df.empty:
        logger.warning(
            "DataFrame vazio após leitura de '%s'. Pulando processamento.", source_key
        )
        return False

    row_count, col_count = df.shape
    logger.info("Arquivo '%s': %d linhas x %d colunas.", source_key, row_count, col_count)

    # Calcula proporção de nulos por coluna e loga as que ultrapassam o limiar
    null_ratios = df.isnull().mean()
    high_null_cols = null_ratios[null_ratios > NULL_THRESHOLD]

    if not high_null_cols.empty:
        for col, ratio in high_null_cols.items():
            logger.warning(
                "Coluna '%s' em '%s' tem %.1f%% de valores nulos.",
                col, source_key, ratio * 100
            )

    return True


# ============================================================
# Funções auxiliares – Glue Data Catalog
# ============================================================

def create_database_if_not_exists(database_name: str) -> None:
    """
    Cria o database no Glue Data Catalog caso ainda não exista.
    Operação idempotente: não falha se o database já existir.
    """
    try:
        glue.get_database(Name=database_name)
        logger.info("Database já existe: %s", database_name)
    except glue.exceptions.EntityNotFoundException:
        glue.create_database(
            DatabaseInput={
                "Name": database_name,
                "Description": "Database do case de riscos criado via Glue Script"
            }
        )
        logger.info("Database criado: %s", database_name)


# ============================================================
# Funções auxiliares – Escrita na Silver e Catalogação
# ============================================================

def write_dataframe_to_silver(
    df_pandas: pd.DataFrame,
    table_name: str,
    source_file_key: str
) -> str:
    """
    Converte um DataFrame Pandas para Spark DataFrame, adiciona colunas de
    metadados/lineage e grava em Parquet comprimido (Snappy) na camada Silver.

    Parâmetros:
        df_pandas       -- DataFrame com os dados da Bronze (colunas já normalizadas)
        table_name      -- Nome da tabela destino (usado como prefixo S3)
        source_file_key -- Chave S3 do arquivo de origem (para rastreabilidade)

    Retorna o caminho S3 completo onde os dados foram gravados.
    """
    # Garante que todas as colunas estejam normalizadas antes de converter para Spark
    df_pandas = normalize_dataframe_columns(df_pandas)

    # Converte todas as colunas para string e substitui NaN por string vazia.
    # Isso evita erros de inferência de tipo ao criar o schema Spark dinamicamente
    # e garante compatibilidade homogênea com Athena (schema all-string).
    df_pandas = df_pandas.astype(str).fillna("")

    # Cria o Spark DataFrame a partir do Pandas — o schema é inferido automaticamente
    df_spark = spark.createDataFrame(df_pandas)

    # Adiciona colunas de metadados para observabilidade e rastreabilidade (lineage):
    #   data_processamento  : timestamp UTC do momento de escrita na Silver
    #   camada_origem       : indica que os dados vieram da camada Bronze
    #   nome_arquivo_origem : chave S3 exata do arquivo que originou esta carga
    df_spark = (
        df_spark
        .withColumn("data_processamento", current_timestamp())
        .withColumn("camada_origem", lit("bronze"))
        .withColumn("nome_arquivo_origem", lit(source_file_key))
    )

    output_path = f"s3://{BUCKET}/{SILVER_PREFIX}/{table_name}/"

    (
        df_spark
        .write
        .mode("overwrite")                    # Sobrescreve partição completa — garante idempotência
        .format("parquet")
        .option("compression", "snappy")      # Snappy: melhor equilíbrio velocidade/tamanho para Athena
        .save(output_path)
    )

    logger.info("Tabela gravada na Silver: %s -> %s", table_name, output_path)
    return output_path


def catalog_table(table_name: str, s3_path: str, df_pandas: pd.DataFrame) -> None:
    """
    Cria ou atualiza a definição da tabela externa no Glue Data Catalog,
    apontando para os arquivos Parquet gravados na camada Silver.
    Operação idempotente: usa update_table se já existe, create_table caso contrário.
    """
    # Monta lista de colunas de negócio a partir do DataFrame já normalizado
    columns = [
        {"Name": normalize_column_name(col), "Type": "string"}
        for col in df_pandas.columns
    ]

    # Adiciona as colunas de metadados geradas em write_dataframe_to_silver.
    # Devem refletir exatamente o schema do Parquet para que o Athena leia corretamente.
    columns.append({"Name": "data_processamento",  "Type": "timestamp"})
    columns.append({"Name": "camada_origem",        "Type": "string"})
    columns.append({"Name": "nome_arquivo_origem",  "Type": "string"})

    table_input = {
        "Name": table_name,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {
            "classification": "parquet",
            "typeOfData": "file"
        },
        "StorageDescriptor": {
            "Columns": columns,
            # Aponta para o prefixo S3 onde os arquivos Parquet foram gravados
            "Location": s3_path,
            # Formatos de I/O padrão do Hive para Parquet — necessários para Athena
            "InputFormat":  "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                # SerDe (Serializer/Deserializer) responsável por ler/escrever colunas Parquet
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                "Parameters": {
                    # Parâmetro obrigatório pelo SerDe Parquet do Hive
                    "serialization.format": "1"
                }
            }
        }
    }

    try:
        # Verifica se a tabela já existe — não há upsert nativo no boto3 Glue
        glue.get_table(DatabaseName=DATABASE_NAME, Name=table_name)
        # Tabela existe: atualiza para refletir possíveis mudanças de colunas
        glue.update_table(DatabaseName=DATABASE_NAME, TableInput=table_input)
        logger.info("Tabela atualizada no catálogo: %s", table_name)
    except glue.exceptions.EntityNotFoundException:
        # Tabela não existe: cria pela primeira vez
        glue.create_table(DatabaseName=DATABASE_NAME, TableInput=table_input)
        logger.info("Tabela criada no catálogo: %s", table_name)


# ============================================================
# Função principal de processamento por pasta
# ============================================================

def process_source_folder(folder_name: str) -> None:
    """
    Orquestra o processamento completo de uma pasta da camada Bronze:
        1. Lista arquivos válidos no S3.
        2. Lê cada arquivo CSV/XLSX em um DataFrame Pandas.
        3. Valida a qualidade mínima dos dados.
        4. Grava os dados em Parquet na camada Silver.
        5. Cataloga a tabela no Glue Data Catalog.

    Em caso de falha em um arquivo individual, loga o erro e continua para
    o próximo — evita que um arquivo corrompido interrompa as demais fontes.
    """
    bronze_folder_prefix = f"{BRONZE_PREFIX}/{folder_name}/"
    files = list_files_from_s3_folder(BUCKET, bronze_folder_prefix)

    # Filtra apenas extensões suportadas pelo job
    valid_files = [
        f for f in files
        if f.lower().endswith((".csv", ".xlsx"))
    ]

    if not valid_files:
        logger.warning("Nenhum arquivo válido em: %s", bronze_folder_prefix)
        return

    # O nome da tabela é derivado do nome da pasta, garantindo consistência
    table_name = normalize_column_name(folder_name)

    for file_key in valid_files:
        logger.info("Iniciando processamento: %s", file_key)
        start_time = time.perf_counter()  # Marca início para medir duração total do arquivo

        try:
            df_pandas = read_s3_file_to_pandas(BUCKET, file_key)

            # Valida qualidade mínima; pula o arquivo se inválido (ex.: vazio)
            if not validate_dataframe(df_pandas, file_key):
                continue

            silver_path = write_dataframe_to_silver(df_pandas, table_name, file_key)

            # Re-normaliza as colunas do Pandas para alinhar com o schema gravado no Parquet
            df_pandas = normalize_dataframe_columns(df_pandas)
            catalog_table(table_name, silver_path, df_pandas)

        except Exception as exc:
            # Isola falhas por arquivo: loga traceback completo mas não aborta o job
            logger.error(
                "Falha ao processar '%s': %s",
                file_key, exc,
                exc_info=True
            )
            continue

        elapsed = time.perf_counter() - start_time
        logger.info("Arquivo processado em %.2fs: %s", elapsed, file_key)


# ============================================================
# Execução principal
# ============================================================

job_start = time.perf_counter()
logger.info("Iniciando job: %s", args["JOB_NAME"])

create_database_if_not_exists(DATABASE_NAME)

for folder in SOURCE_FOLDERS:
    logger.info("=== Processando pasta: %s ===", folder)
    process_source_folder(folder)

job.commit()

total_elapsed = time.perf_counter() - job_start
logger.info("Job finalizado em %.2fs.", total_elapsed)