import sys
import time
import logging
from typing import Tuple

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, lower, trim, current_timestamp, countDistinct,
    count, avg, when, regexp_replace
)


# ============================================================
# Configuração de Logging
# ============================================================

# Logger estruturado; no AWS Glue as mensagens são encaminhadas ao CloudWatch Logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuração do Job
# ============================================================

# --- Identificação do database e caminho de saída ---
DATABASE   = "workspace_db_case_espec_dados_riscos"
GOLD_PATH  = "s3://workspace-db-case-espec-dados-riscos/gold/gold_risco_tecnologico_por_sigla/"
GOLD_TABLE = "gold_risco_tecnologico_por_sigla"

# --- Pesos do score de risco operacional (altere aqui para recalibrar o modelo) ---
PESO_RISCO_ALTO_MEDIO = 3.0
PESO_IMPACTO_CRITICO  = 2.0
PESO_OBSOLESCENCIA    = 2.0
PESO_SERVIDOR_PROD    = 0.2

# --- Limiares de classificação do score de risco (inclusive) ---
LIMIAR_RISCO_ALTO  = 50.0
LIMIAR_RISCO_MEDIO = 20.0


# ============================================================
# Funções puras de transformação
# ============================================================

def normalize_text(column_name: str):
    """
    Retorna uma expressão Spark que aplica lower() + trim() a uma coluna.
    Usado para padronizar chaves de join e evitar falhas por diferença de caixa ou espaços.
    """
    return lower(trim(col(column_name)))


def read_silver_tables(
    spark: SparkSession, database: str
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Lê as quatro tabelas da camada Silver do Glue Data Catalog.
    Retorna uma tupla (riscos, servidores, softwares, pacotes).
    Lança AnalysisException se qualquer tabela não existir no catálogo.
    """
    def _read(table_name: str) -> DataFrame:
        full_name = f"{database}.{table_name}"
        logger.info("Lendo tabela do catálogo: %s", full_name)
        return spark.table(full_name)

    return (
        _read("resultado_query3"),
        _read("servidores_siglas"),
        _read("cmdb_software_instance_sot"),
        _read("cmdb_ci_spkg_sot"),
    )


def treat_risks(df: DataFrame) -> DataFrame:
    """
    Aplica limpeza e enriquecimento de flags na base de riscos.
    Entrada: DataFrame bruto da tabela resultado_query3.
    Saída: DataFrame com chave de join, índices numéricos e flags binárias.
    """
    return (
        df
        # Cria chave de join padronizada (minúsculo + sem espaços) a partir da sigla do sistema
        .withColumn("sigla_join", normalize_text("sigla_ss"))

        # Normaliza separador decimal de vírgula para ponto antes de converter para double,
        # pois o campo pode vir como string "0,75" de fontes brasileiras
        .withColumn(
            "indice_obsolescencia_num",
            regexp_replace(col("indice_obsolescencia"), ",", ".").cast("double")
        )
        .withColumn(
            "criticidade_num",
            regexp_replace(col("criticidade"), ",", ".").cast("double")
        )

        # Flag binária: 1 se o rating indica risco alto ou médio (inclui variações sem acento)
        .withColumn(
            "flag_risco_alto",
            when(lower(col("rating")).isin("alto", "médio", "medio"), 1).otherwise(0)
        )

        # Flag binária: 1 se o campo impacto contém "cr" (prefixo de "crítico")
        .withColumn(
            "flag_critico",
            when(lower(col("impacto")).contains("cr"), 1).otherwise(0)
        )

        # Flag binária: 1 se o item é hospedado em cloud (aceita variações textuais comuns)
        .withColumn(
            "flag_cloud",
            when(lower(col("cloud")).isin("true", "verdadeiro", "sim"), 1).otherwise(0)
        )
    )


def aggregate_risks(df: DataFrame) -> DataFrame:
    """
    Agrega métricas de risco tecnológico por sigla.
    Entrada: DataFrame enriquecido por treat_risks().
    Saída: uma linha por sigla com contagens e médias.
    """
    return (
        df
        .groupBy("sigla_join")
        .agg(
            # Produtos tecnológicos distintos vinculados à sigla
            countDistinct("produto").alias("qtd_produtos_tecnologicos"),

            # Total de relacionamentos (linhas) para esta sigla
            count("*").alias("qtd_relacionamentos_tecnologia"),

            # Médias numéricas — NULLs são ignorados pelo avg() do Spark
            avg("indice_obsolescencia_num").alias("media_indice_obsolescencia"),
            avg("criticidade_num").alias("media_criticidade"),

            # Contagem condicional: when() atua como filtro dentro do count()
            count(when(col("flag_risco_alto") == 1, True)).alias("qtd_riscos_altos_medios"),
            count(when(col("flag_critico")    == 1, True)).alias("qtd_impactos_criticos"),
            count(when(col("flag_cloud")      == 1, True)).alias("qtd_itens_cloud"),
        )
    )


def aggregate_servers(
    df: DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """
    Normaliza servidores e agrega métricas de infraestrutura por sigla.
    Retorna uma tupla (servidores_tratado, servidores_por_sigla).
    servidores_tratado é necessário como entrada para aggregate_softwares().
    """
    # Adiciona chave de join padronizada para cruzamento com a tabela de riscos
    servidores_tratado = df.withColumn("sigla_join", normalize_text("sigla"))

    servidores_por_sigla = (
        servidores_tratado
        .groupBy("sigla_join")
        .agg(
            # Servidores únicos (pelo nome) associados à sigla
            countDistinct("nome").alias("qtd_servidores"),

            # Diversidade de arquiteturas (x86, ARM, etc.) como proxy de heterogeneidade
            countDistinct("arquitetura").alias("qtd_tipos_arquitetura"),

            # contains("produ") captura "producao", "produção", "prod", etc.
            count(when(lower(col("ambiente")).contains("produ"), True)).alias("qtd_servidores_producao"),

            # Comparação exata após lower() para status operacional ativo
            count(when(lower(col("status_operacional")) == "operacional", True)).alias("qtd_servidores_operacionais"),
        )
    )

    return servidores_tratado, servidores_por_sigla


def aggregate_softwares(
    softwares_df: DataFrame,
    servidores_tratado: DataFrame,
) -> DataFrame:
    """
    Calcula a média de softwares instalados por sigla.
    Faz join entre softwares (chave: installed_on) e servidores (chave: nome)
    para depois agregar no nível de sigla.
    Left join garante que servidores sem softwares cadastrados sejam mantidos.
    """
    # Agrega softwares pelo servidor onde estão instalados, normalizando a chave de join
    softwares_por_servidor = (
        softwares_df
        .groupBy(lower(trim(col("installed_on"))).alias("nome_servidor_join"))
        .agg(countDistinct("software").alias("qtd_softwares_instalados"))
    )

    return (
        servidores_tratado
        .withColumn("nome_servidor_join", normalize_text("nome"))
        .join(softwares_por_servidor, "nome_servidor_join", "left")
        .groupBy("sigla_join")
        .agg(avg("qtd_softwares_instalados").alias("media_softwares_por_servidor"))
    )


def build_gold(
    risco_por_sigla: DataFrame,
    servidores_por_sigla: DataFrame,
    softwares_por_sigla: DataFrame,
) -> DataFrame:
    """
    Constrói a tabela Gold unindo as três visões agregadas por sigla
    e calculando o score e a classificação de risco operacional.
    A tabela de riscos é a âncora: toda sigla com risco é mantida (left joins).
    Os pesos e limiares são controlados pelas constantes PESO_* e LIMIAR_*.
    """
    return (
        risco_por_sigla
        .join(servidores_por_sigla, "sigla_join", "left")
        .join(softwares_por_sigla,  "sigla_join", "left")

        # Renomeia a chave técnica para o nome de negócio final
        .withColumnRenamed("sigla_join", "sigla")

        # Score composto ponderado — combina múltiplas dimensões de risco
        .withColumn(
            "score_risco_operacional",
            col("qtd_riscos_altos_medios")     * PESO_RISCO_ALTO_MEDIO
            + col("qtd_impactos_criticos")      * PESO_IMPACTO_CRITICO
            + col("media_indice_obsolescencia") * PESO_OBSOLESCENCIA
            + col("qtd_servidores_producao")    * PESO_SERVIDOR_PROD
        )

        # Classificação baseada nos limiares configuráveis
        .withColumn(
            "classificacao_risco",
            when(col("score_risco_operacional") >= LIMIAR_RISCO_ALTO,  "Alto")
            .when(col("score_risco_operacional") >= LIMIAR_RISCO_MEDIO, "Médio")
            .otherwise("Baixo")
        )

        # Timestamp de processamento para auditoria e rastreabilidade na camada Gold
        .withColumn("data_processamento", current_timestamp())
    )


def write_gold(df: DataFrame, database: str, table: str, path: str) -> None:
    """
    Persiste o DataFrame Gold em Parquet comprimido (Snappy) no S3
    e registra/atualiza a entrada no Glue Data Catalog via saveAsTable.
    Operação idempotente: mode("overwrite") garante reprocessamento seguro.
    """
    (
        df.write
        .mode("overwrite")               # Sobrescreve completamente — garante idempotência
        .format("parquet")
        .option("compression", "snappy") # Snappy: melhor equilíbrio velocidade/tamanho para Athena
        .option("path", path)            # Grava fisicamente no prefixo S3 definido
        .saveAsTable(f"{database}.{table}")  # Registra/atualiza entrada no Glue Data Catalog
    )


def main(spark: SparkSession, job_name: str) -> None:
    """
    Orquestra o pipeline Silver → Gold como composição de funções puras.

    Fluxo:
        read_silver_tables
            → treat_risks → aggregate_risks         ┐
            → aggregate_servers                      ├─→ build_gold → write_gold
            → aggregate_softwares                    ┘
    """
    logger.info("Iniciando job: %s", job_name)
    job_start = time.perf_counter()

    # --- Fase 1: Leitura ---
    logger.info("=== Fase 1: Leitura das tabelas Silver ===")
    try:
        riscos, servidores, softwares, _ = read_silver_tables(spark, DATABASE)
    except Exception as exc:
        logger.error("Falha na leitura das tabelas Silver: %s", exc, exc_info=True)
        raise

    # --- Fases 2–5: Pipeline de transformações (composição funcional) ---
    logger.info("=== Fases 2–5: Transformações ===")
    _t = time.perf_counter()

    # Cada função recebe um DataFrame e devolve um DataFrame — sem efeitos colaterais
    risco_por_sigla                      = aggregate_risks(treat_risks(riscos))
    servidores_tratado, servidores_por_sigla = aggregate_servers(servidores)
    softwares_por_sigla                  = aggregate_softwares(softwares, servidores_tratado)

    # Composição final: une as três visões e aplica o modelo de score
    gold = build_gold(risco_por_sigla, servidores_por_sigla, softwares_por_sigla)

    logger.info("Pipeline de transformações definido em %.2fs.", time.perf_counter() - _t)

    # --- Fase 6: Escrita ---
    logger.info("=== Fase 6: Escrita na camada Gold ===")
    _t = time.perf_counter()

    try:
        write_gold(gold, DATABASE, GOLD_TABLE, GOLD_PATH)
    except Exception as exc:
        logger.error("Falha ao gravar tabela Gold '%s': %s", GOLD_TABLE, exc, exc_info=True)
        raise

    logger.info(
        "Tabela Gold '%s' gravada em %.2fs. Caminho: %s",
        GOLD_TABLE, time.perf_counter() - _t, GOLD_PATH
    )
    logger.info("Job finalizado em %.2fs.", time.perf_counter() - job_start)


# ============================================================
# Inicialização do Glue Job e execução
# ============================================================

# Lê parâmetros injetados pelo Glue no momento da execução do job
args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

main(spark, args["JOB_NAME"])

job.commit()
