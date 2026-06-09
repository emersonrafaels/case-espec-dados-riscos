import sys
import time
import logging

from pyspark.context import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions


# ============================================================
# Configuração de Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuração do Job
# ============================================================

DATABASE_NAME = "workspace_db_case_espec_dados_riscos"
BUCKET = "workspace-db-case-espec-dados-riscos"

# Fonte principal criada no job_03, com nomenclatura padronizada
SOURCE_GOLD_TABLE = "gold_obsolescencia_por_servidor"

# Outputs finais exigidos pelo case; ambos terminam em _por_servidor
GOLD_HUB_TABLE = "gold_hub_analitico_por_servidor"
GOLD_CONTROLE_TABLE = "gold_controle_inconformidades_por_servidor"

GOLD_HUB_PATH = f"s3://{BUCKET}/gold/{GOLD_HUB_TABLE}/"
GOLD_CONTROLE_PATH = f"s3://{BUCKET}/gold/{GOLD_CONTROLE_TABLE}/"

# Farol do case
FAROL_VERDE_MAX = 5.0
FAROL_AMARELO_MAX = 20.0


# ============================================================
# Leitura
# ============================================================

def read_gold_por_servidor(spark: SparkSession) -> DataFrame:
    """
    Lê a Gold operacional criada no job_03, já na granularidade por servidor.
    """
    full_name = f"{DATABASE_NAME}.{SOURCE_GOLD_TABLE}"
    logger.info("Lendo tabela fonte: %s", full_name)
    return spark.table(full_name)


# ============================================================
# Construção dos outputs finais do case
# ============================================================

def build_hub_analitico_por_servidor(df: DataFrame) -> DataFrame:
    """
    Gera o arquivo final no mesmo layout do HUB Analítico do case.

    Colunas obrigatórias:
        INDICADOR, Nome_Catch, Chave, Tipo_chave, Descricao_chave,
        Criticidade, Diretoria, Comunidade, Conformidade,
        Nome_Cabecalho, Detalhe_Cabecalho.
    """
    return (
        df.withColumn(
            "Conformidade",
            F.when(
                (F.col("indice_obsolescencia") > 0)
                | F.lower(F.col("rating")).isin("alto", "médio", "medio", "baixo"),
                F.lit("Não esta em conformidade"),
            ).otherwise(F.lit("Em conformidade")),
        )
        .withColumn(
            "Detalhe_Cabecalho_Final",
            F.concat_ws(
                " | ",
                F.concat(F.lit("sigla_origem="), F.coalesce(F.col("sigla_origem"), F.lit(""))),
                F.concat(F.lit("software_instalado="), F.coalesce(F.col("software_instalado"), F.lit(""))),
                F.concat(F.lit("software_catalogado="), F.coalesce(F.col("software_catalogado"), F.lit(""))),
                F.concat(F.lit("produto_tecnologico="), F.coalesce(F.col("produto_tecnologico"), F.lit(""))),
                F.concat(F.lit("versao="), F.coalesce(F.col("versao"), F.lit(""))),
                F.concat(F.lit("status="), F.coalesce(F.col("status_relacionamento"), F.lit(""))),
                F.concat(F.lit("criticidade="), F.coalesce(F.col("criticidade").cast("string"), F.lit(""))),
                F.concat(F.lit("indice_obsolescencia="), F.coalesce(F.col("indice_obsolescencia").cast("string"), F.lit(""))),
                F.concat(F.lit("rating="), F.coalesce(F.col("rating"), F.lit(""))),
                F.concat(F.lit("impacto="), F.coalesce(F.col("impacto"), F.lit(""))),
                F.concat(F.lit("cloud="), F.coalesce(F.col("cloud"), F.lit(""))),
                F.concat(F.lit("ambiente="), F.coalesce(F.col("ambiente"), F.lit(""))),
                F.concat(F.lit("arquitetura="), F.coalesce(F.col("arquitetura"), F.lit(""))),
            ),
        )
        .select(
            F.lit("Obsolescência tecnológica por servidor").alias("INDICADOR"),
            F.lit("Controle de softwares obsoletos por servidor").alias("Nome_Catch"),
            F.col("servidor").alias("Chave"),
            F.lit("Servidor").alias("Tipo_chave"),
            F.col("descricao_servidor").alias("Descricao_chave"),
            # Mantemos o campo Criticidade no layout original, usando impacto como proxy operacional
            F.coalesce(F.col("impacto"), F.lit("Não informado")).alias("Criticidade"),
            F.lit("Não informado").alias("Diretoria"),
            F.lit("Não informado").alias("Comunidade"),
            F.col("Conformidade"),
            F.lit(
                "sigla_origem | software_instalado | software_catalogado | produto_tecnologico | "
                "versao | status | criticidade | indice_obsolescencia | rating | impacto | cloud | ambiente | arquitetura"
            ).alias("Nome_Cabecalho"),
            F.col("Detalhe_Cabecalho_Final").alias("Detalhe_Cabecalho"),
        )
        .dropDuplicates()
    )


def build_controle_inconformidades_por_servidor(hub_df: DataFrame) -> DataFrame:
    """
    Gera o percentual de inconformidades e farol do case.

    Regra:
        Verde    <= 5%
        Amarelo  > 5% e < 20%
        Vermelho >= 20%
    """
    return (
        hub_df.agg(
            F.count("*").alias("total_registros"),
            F.sum(
                F.when(F.col("Conformidade") == "Não esta em conformidade", 1).otherwise(0)
            ).alias("total_inconformes"),
        )
        .withColumn(
            "percentual_inconformidade",
            F.round((F.col("total_inconformes") / F.col("total_registros")) * 100, 2),
        )
        .withColumn(
            "farol",
            F.when(F.col("percentual_inconformidade") <= FAROL_VERDE_MAX, F.lit("Verde"))
            .when(F.col("percentual_inconformidade") < FAROL_AMARELO_MAX, F.lit("Amarelo"))
            .otherwise(F.lit("Vermelho")),
        )
        .withColumn(
            "criterio_farol",
            F.lit("Verde <= 5%; Amarelo > 5% e < 20%; Vermelho >= 20%"),
        )
        .withColumn("data_processamento", F.current_timestamp())
    )


# ============================================================
# Escrita
# ============================================================

def write_gold_table(spark: SparkSession, df: DataFrame, table_name: str, path: str) -> None:
    """
    Grava uma tabela Gold em Parquet e registra no Glue Data Catalog.
    """
    full_name = f"{DATABASE_NAME}.{table_name}"
    logger.info("Recriando tabela: %s", full_name)
    spark.sql(f"DROP TABLE IF EXISTS {full_name}")

    (
        df.write.mode("overwrite")
        .format("parquet")
        .option("compression", "snappy")
        .option("path", path)
        .saveAsTable(full_name)
    )

    logger.info("Tabela gravada: %s -> %s", full_name, path)


# ============================================================
# Execução principal
# ============================================================

def main(spark: SparkSession, job_name: str) -> None:
    logger.info("Iniciando job: %s", job_name)
    start = time.perf_counter()

    gold_por_servidor = read_gold_por_servidor(spark)

    hub_por_servidor = build_hub_analitico_por_servidor(gold_por_servidor)
    controle_por_servidor = build_controle_inconformidades_por_servidor(hub_por_servidor)

    write_gold_table(spark, hub_por_servidor, GOLD_HUB_TABLE, GOLD_HUB_PATH)
    write_gold_table(spark, controle_por_servidor, GOLD_CONTROLE_TABLE, GOLD_CONTROLE_PATH)

    logger.info("Job finalizado em %.2fs", time.perf_counter() - start)


args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(args["JOB_NAME"], args)

main(spark, args["JOB_NAME"])

job.commit()
