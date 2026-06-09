import sys
import time
import logging

from pyspark.context import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


DATABASE_NAME = "workspace_db"
BUCKET = "workspace-db-case-espec-dados-riscos"

SOURCE_GOLD_TABLE = "case_riscos_gold_obsolescencia_por_servidor"
DIM_SIGLAS_TABLE = "case_riscos_resultado_query1"
DIM_PRODUTOS_TABLE = "case_riscos_resultado_query2"

GOLD_HUB_TABLE = "case_riscos_gold_hub_analitico_por_servidor"
GOLD_CONTROLE_TABLE = "case_riscos_gold_controle_inconformidades_por_servidor"

GOLD_HUB_PATH = f"s3://{BUCKET}/case-riscos/gold/{GOLD_HUB_TABLE}/"
GOLD_CONTROLE_PATH = f"s3://{BUCKET}/case-riscos/gold/{GOLD_CONTROLE_TABLE}/"

FAROL_VERDE_MAX = 5.0
FAROL_AMARELO_MAX = 20.0

# --- Particionamento ---
ANOMES_DEFAULT = "202604"  # Valor padrão para a partição anomes (AAAAMM)


def normalize_col(column_name: str):
    return F.lower(F.trim(F.col(column_name)))


def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    full_name = f"{DATABASE_NAME}.{table_name}"
    logger.info("Lendo tabela: %s", full_name)
    return spark.table(full_name)


def prepare_dim_siglas(df: DataFrame) -> DataFrame:
    """
    Prepara Resultado_QUERY1:
    sigla_ss, descricao_da_sigla, comunidade, diretoria, criticidade_final.
    """
    return (
        df.select(
            normalize_col("sigla_ss").alias("sigla_join"),
            F.col("criticidade_final").alias("criticidade_tier"),
            F.col("diretoria").alias("diretoria"),
            F.col("comunidade").alias("comunidade"),
        )
        .filter(F.col("sigla_join").isNotNull())
        .dropDuplicates(["sigla_join"])
    )


def prepare_dim_produtos(df: DataFrame) -> DataFrame:
    """
    Prepara Resultado_QUERY2:
    nome, pillar, dominio, subdominio.
    """
    return (
        df.select(
            normalize_col("nome").alias("produto_join"),
            F.col("nome").alias("nome_produto"),
            F.col("pillar").alias("pillar"),
            F.col("dominio").alias("dominio"),
            F.col("subdominio").alias("subdominio"),
        )
        .filter(F.col("produto_join").isNotNull())
        .dropDuplicates(["produto_join"])
    )


def prepare_gold_por_servidor(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("sigla_join", normalize_col("sigla_origem"))
        .withColumn(
            "produto_join",
            F.lower(
                F.trim(
                    F.coalesce(
                        F.col("produto_tecnologico"),
                        F.col("software_catalogado"),
                        F.col("software_instalado"),
                    )
                )
            ),
        )
    )


def build_hub_analitico_por_servidor(
    gold_df: DataFrame,
    dim_siglas: DataFrame,
    dim_produtos: DataFrame,
) -> DataFrame:
    return (
        gold_df.alias("gold")
        .join(dim_siglas.alias("sig"), on="sigla_join", how="left")
        .join(dim_produtos.alias("prod"), on="produto_join", how="left")
        .withColumn(
            "Conformidade",
            F.when(
                (F.col("gold.indice_obsolescencia") > 0)
                | F.lower(F.col("gold.rating")).isin("alto", "médio", "medio", "baixo"),
                F.lit("Não esta em conformidade"),
            ).otherwise(F.lit("Em conformidade")),
        )
        .withColumn(
            "Detalhe_Cabecalho_Final",
            F.concat_ws(
                " | ",
                F.coalesce(F.col("gold.produto_tecnologico"), F.col("gold.software_catalogado"), F.col("gold.software_instalado"), F.lit("")),
                F.coalesce(F.col("gold.versao"), F.lit("")),
                F.coalesce(F.col("gold.status_relacionamento"), F.lit("")),
                F.coalesce(F.col("gold.criticidade").cast("string"), F.lit("")),
                F.coalesce(F.col("gold.indice_obsolescencia").cast("string"), F.lit("")),
                F.coalesce(F.col("gold.rating"), F.lit("")),
                F.coalesce(F.col("gold.impacto"), F.lit("")),
                F.coalesce(F.col("gold.cloud"), F.lit("")),
                F.coalesce(F.col("prod.nome_produto"), F.lit("")),
                F.coalesce(F.col("prod.pillar"), F.lit("")),
                F.coalesce(F.col("prod.dominio"), F.lit("")),
                F.coalesce(F.col("prod.subdominio"), F.lit("")),
                F.concat(F.lit("sigla_origem="), F.coalesce(F.col("gold.sigla_origem"), F.lit(""))),
                F.concat(F.lit("ambiente="), F.coalesce(F.col("gold.ambiente"), F.lit(""))),
                F.concat(F.lit("arquitetura="), F.coalesce(F.col("gold.arquitetura"), F.lit(""))),
            ),
        )
        .select(
            F.lit("Obsolescência tecnológica por servidor").alias("INDICADOR"),
            F.lit("Controle de softwares obsoletos por servidor").alias("Nome_Catch"),
            F.col("gold.servidor").alias("Chave"),
            F.lit("Servidor").alias("Tipo_chave"),
            F.col("gold.descricao_servidor").alias("Descricao_chave"),
            F.coalesce(F.col("sig.criticidade_tier"), F.lit("Não informado")).alias("Criticidade"),
            F.coalesce(F.col("sig.diretoria"), F.lit("Não informado")).alias("Diretoria"),
            F.coalesce(F.col("sig.comunidade"), F.lit("Não informado")).alias("Comunidade"),
            F.col("Conformidade"),
            F.lit(
                "produto | versao | status | criticidade | indice_obsolescencia | "
                "rating | impacto | cloud | Nome | Pillar | Dominio | Subdominio | "
                "sigla_origem | ambiente | arquitetura"
            ).alias("Nome_Cabecalho"),
            F.col("Detalhe_Cabecalho_Final").alias("Detalhe_Cabecalho"),
            # Partição por período de referência (AAAAMM) — permite reprocessamento incremental
            F.lit(ANOMES_DEFAULT).alias("anomes"),
        )
        .dropDuplicates()
    )


def build_controle_inconformidades_por_servidor(hub_df: DataFrame) -> DataFrame:
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
        # Partição por período de referência (AAAAMM) — permite reprocessamento incremental
        .withColumn("anomes", F.lit(ANOMES_DEFAULT))
    )


def write_gold_table(spark: SparkSession, df: DataFrame, table_name: str, path: str) -> None:
    full_name = f"{DATABASE_NAME}.{table_name}"
    logger.info("Recriando tabela: %s", full_name)
    spark.sql(f"DROP TABLE IF EXISTS {full_name}")

    (
        df.write.mode("overwrite")
        .format("parquet")
        .option("compression", "snappy")
        .option("path", path)
        .partitionBy("anomes")         # Particiona por período de referência (AAAAMM)
        .saveAsTable(full_name)
    )

    logger.info("Tabela gravada: %s -> %s", full_name, path)


def main(spark: SparkSession, job_name: str) -> None:
    logger.info("Iniciando job: %s", job_name)
    start = time.perf_counter()

    gold_por_servidor = prepare_gold_por_servidor(
        read_table(spark, SOURCE_GOLD_TABLE)
    )

    dim_siglas = prepare_dim_siglas(
        read_table(spark, DIM_SIGLAS_TABLE)
    )

    dim_produtos = prepare_dim_produtos(
        read_table(spark, DIM_PRODUTOS_TABLE)
    )

    hub_por_servidor = build_hub_analitico_por_servidor(
        gold_por_servidor,
        dim_siglas,
        dim_produtos,
    )

    controle_por_servidor = build_controle_inconformidades_por_servidor(
        hub_por_servidor
    )

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