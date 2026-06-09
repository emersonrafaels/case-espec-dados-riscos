import sys
import time
import logging
from typing import Tuple

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


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

DATABASE_NAME = "workspace_db"
BUCKET_NAME = "workspace-db-case-espec-dados-riscos"

# Nome padronizado: sempre termina em _por_servidor
GOLD_TABLE = "case_riscos_gold_obsolescencia_por_servidor"
GOLD_PATH = f"s3://{BUCKET_NAME}/case-riscos/gold/{GOLD_TABLE}/"

# Views analíticas compatíveis com a visão por sigla, mas na granularidade servidor
VW_RANKING_RISCO_POR_SERVIDOR = "case_riscos_vw_ranking_risco_por_servidor"
VW_DETALHE_OBSOLESCENCIA_POR_SERVIDOR = "case_riscos_vw_detalhe_obsolescencia_por_servidor"
VW_RESUMO_RISCO_POR_SERVIDOR = "case_riscos_vw_resumo_risco_por_servidor"

# Pesos do score por servidor
PESO_OBSOLESCENCIA_SERVIDOR = 10.0
PESO_CRITICIDADE_SERVIDOR = 10.0
PESO_RISCO_ALTO_MEDIO = 2.0
PESO_IMPACTO_CRITICO = 3.0

# Limiares do score por servidor
LIMIAR_RISCO_ALTO = 60.0
LIMIAR_RISCO_MEDIO = 20.0

# --- Particionamento ---
ANOMES_DEFAULT = "202604"  # Valor padrão para a partição anomes (AAAAMM)


# ============================================================
# Funções de transformação de colunas
# ============================================================

def normalize_text(column: str):
    """
    Retorna expressão Spark: lower(trim(col)).
    Usado para padronizar chaves de join.
    """
    return F.lower(F.trim(F.col(column)))


def to_double_from_ptbr(column: str):
    """
    Converte número em formato string PT-BR para double.
    Exemplo: '1,25' -> 1.25.
    """
    return F.regexp_replace(F.trim(F.col(column)), ",", ".").cast(DoubleType())


def clean_empty_string(column: str):
    """
    Padroniza valores vazios, nulos, 'nan' e 'empty string' como null.
    """
    return (
        F.when(
            F.col(column).isNull()
            | (F.lower(F.trim(F.col(column))) == "nan")
            | (F.lower(F.trim(F.col(column))) == "empty string")
            | (F.trim(F.col(column)) == ""),
            None,
        )
        .otherwise(F.trim(F.col(column)))
    )


# ============================================================
# Leitura e tratamento
# ============================================================

def read_silver_tables(
    spark: SparkSession, database: str
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Lê as quatro tabelas Silver do Glue Data Catalog.
    Retorna: servidores, relacionamentos, software_instance, software_catalogo.
    """
    def _read(table_name: str) -> DataFrame:
        full_name = f"{database}.{table_name}"
        logger.info("Lendo tabela do catálogo: %s", full_name)
        return spark.table(full_name)

    return (
        _read("case_riscos_servidores_siglas"),
        _read("case_riscos_resultado_query3"),
        _read("case_riscos_cmdb_software_instance_sot"),
        _read("case_riscos_cmdb_ci_spkg_sot"),
    )


def treat_servers(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza servidores, criando chaves para join com software e sigla.
    """
    return (
        df.select(
            clean_empty_string("nome").alias("servidor"),
            clean_empty_string("arquitetura").alias("arquitetura"),
            clean_empty_string("status").alias("status_servidor"),
            clean_empty_string("status_operacional").alias("status_operacional"),
            clean_empty_string("ambiente").alias("ambiente"),
            clean_empty_string("sigla").alias("sigla_origem"),
        )
        .withColumn("servidor_join", F.lower(F.trim(F.col("servidor"))))
        .withColumn("sigla_join", F.lower(F.trim(F.col("sigla_origem"))))
    )


def treat_software_instances(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza instâncias de software instaladas em servidores.
    """
    return (
        df.select(
            clean_empty_string("installed_on").alias("servidor_instancia"),
            clean_empty_string("software").alias("software_instalado"),
            clean_empty_string("anomesdia").alias("anomesdia_software_instance"),
        )
        .withColumn("servidor_join", F.lower(F.trim(F.col("servidor_instancia"))))
        .withColumn("software_join", F.lower(F.trim(F.col("software_instalado"))))
    )


def treat_software_catalog(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza catálogo CMDB de pacotes de software.
    """
    return (
        df.select(
            clean_empty_string("model_id").alias("model_id"),
            clean_empty_string("name").alias("nome_software_catalogo"),
            clean_empty_string("version").alias("versao_catalogo"),
            clean_empty_string("anomesdia").alias("anomesdia_catalogo"),
        )
        .withColumn("model_join", F.lower(F.trim(F.col("model_id"))))
        .withColumn("name_join", F.lower(F.trim(F.col("nome_software_catalogo"))))
    )


def treat_relationships(df: DataFrame) -> DataFrame:
    """
    Limpa relacionamentos de risco tecnológico por sigla/produto.
    """
    return (
        df.select(
            clean_empty_string("sigla_ss").alias("sigla_relacionamento"),
            clean_empty_string("produto").alias("produto_tecnologico"),
            clean_empty_string("versao").alias("versao_relacionamento"),
            clean_empty_string("status").alias("status_relacionamento"),
            to_double_from_ptbr("criticidade").alias("criticidade"),
            to_double_from_ptbr("indice_obsolescencia").alias("indice_obsolescencia"),
            clean_empty_string("rating").alias("rating"),
            clean_empty_string("impacto").alias("impacto"),
            clean_empty_string("cloud").alias("cloud"),
        )
        .withColumn("sigla_join", F.lower(F.trim(F.col("sigla_relacionamento"))))
        .withColumn("produto_join", F.lower(F.trim(F.col("produto_tecnologico"))))
    )


# ============================================================
# Construção da Gold por Servidor
# ============================================================

def build_gold_por_servidor(
    servidores: DataFrame,
    software_instance: DataFrame,
    software_catalogo: DataFrame,
    relacionamentos: DataFrame,
) -> DataFrame:
    """
    Constrói a Gold na granularidade Servidor → Software → Versão → Obsolescência.
    """
    return (
        servidores.alias("srv")
        .join(software_instance.alias("inst"), on="servidor_join", how="left")
        .join(
            software_catalogo.alias("cat"),
            F.col("inst.software_join") == F.col("cat.name_join"),
            how="left",
        )
        .join(
            relacionamentos.alias("rel"),
            (F.col("srv.sigla_join") == F.col("rel.sigla_join"))
            & (
                (F.col("cat.name_join") == F.col("rel.produto_join"))
                | (F.col("inst.software_join") == F.col("rel.produto_join"))
            ),
            how="left",
        )
        .select(
            F.col("srv.servidor").alias("servidor"),
            F.lit("Servidor").alias("tipo_chave"),
            F.col("srv.servidor").alias("descricao_servidor"),
            F.col("srv.sigla_origem"),
            F.col("srv.arquitetura"),
            F.col("srv.status_servidor"),
            F.col("srv.status_operacional"),
            F.col("srv.ambiente"),
            F.col("inst.software_instalado"),
            F.col("cat.nome_software_catalogo").alias("software_catalogado"),
            F.coalesce(F.col("cat.versao_catalogo"), F.col("rel.versao_relacionamento")).alias("versao"),
            F.col("rel.produto_tecnologico"),
            F.col("rel.status_relacionamento"),
            F.col("rel.criticidade"),
            F.col("rel.indice_obsolescencia"),
            F.col("rel.rating"),
            F.col("rel.impacto"),
            F.col("rel.cloud"),
            F.concat_ws(
                " | ",
                F.concat(F.lit("servidor="), F.col("srv.servidor")),
                F.concat(F.lit("sigla_origem="), F.col("srv.sigla_origem")),
                F.concat(F.lit("software="), F.coalesce(F.col("inst.software_instalado"), F.lit("N/A"))),
                F.concat(F.lit("produto_tecnologico="), F.coalesce(F.col("rel.produto_tecnologico"), F.lit("N/A"))),
                F.concat(F.lit("versao="), F.coalesce(F.col("cat.versao_catalogo"), F.col("rel.versao_relacionamento"), F.lit("N/A"))),
            ).alias("detalhe_cabecalho"),
            F.current_timestamp().alias("data_processamento"),
            F.lit("gold").alias("camada"),
            # Partição por período de referência (AAAAMM) — permite reprocessamento incremental
            F.lit(ANOMES_DEFAULT).alias("anomes"),
        )
        .dropDuplicates()
    )


def write_gold(spark: SparkSession, df: DataFrame, database: str, table: str, path: str) -> None:
    """
    Persiste DataFrame como tabela Gold no Glue Data Catalog.

    NÃO executa DROP TABLE antes do write em tabelas particionadas.
    Motivo: DROP TABLE remove a LOCATION do catálogo; saveAsTable subsequente
    tenta ler a localização da entrada do catálogo (agora ausente), obtém string
    vazia e lança 'Can not create a Path from an empty string'.

    Estratégia adotada:
    - partitionOverwriteMode=dynamic  →  mode("overwrite") sobrescreve apenas
      as partições presentes no DataFrame, sem tocar nas demais.
    - Primeira execução (tabela inexistente): saveAsTable cria a tabela no
      catálogo com a LOCATION apontando para `path`.
    - Reexecuções: Spark atualiza apenas a partição anomes correspondente.
    """
    full_name = f"{database}.{table}"
    logger.info("Gravando tabela: %s -> %s", full_name, path)

    # Sobrescreve somente a partição presente no DataFrame (não a tabela inteira)
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    (
        df.write
        .mode("overwrite")
        .format("parquet")
        .option("compression", "snappy")
        .option("path", path)       # LOCATION usada pelo catálogo na primeira execução
        .partitionBy("anomes")      # Partição por período de referência (AAAAMM)
        .saveAsTable(full_name)     # Cria ou atualiza a entrada no Glue Data Catalog
    )

    logger.info("Tabela gravada: %s -> %s", full_name, path)


# ============================================================
# Views analíticas por Servidor
# ============================================================

def create_views_por_servidor(spark: SparkSession, database: str, gold_table: str) -> None:
    """
    Cria views analíticas por servidor, compatíveis com as visões por sigla.
    """
    score_sql = f"""
        COALESCE(AVG(indice_obsolescencia), 0) * {PESO_OBSOLESCENCIA_SERVIDOR}
        + COALESCE(AVG(criticidade), 0) * {PESO_CRITICIDADE_SERVIDOR}
        + COALESCE(SUM(CASE WHEN rating IN ('Alto', 'Médio', 'Medio') THEN 1 ELSE 0 END), 0) * {PESO_RISCO_ALTO_MEDIO}
        + COALESCE(SUM(CASE WHEN impacto IN ('Crítico', 'Critico') THEN 1 ELSE 0 END), 0) * {PESO_IMPACTO_CRITICO}
    """

    spark.sql(f"""
        CREATE OR REPLACE VIEW {database}.{VW_RANKING_RISCO_POR_SERVIDOR} AS
        SELECT
            servidor,
            sigla_origem,
            ambiente,
            arquitetura,
            COUNT(DISTINCT software_instalado) AS qtd_softwares_instalados,
            COUNT(DISTINCT produto_tecnologico) AS qtd_produtos_tecnologicos,
            AVG(indice_obsolescencia) AS media_indice_obsolescencia,
            AVG(criticidade) AS media_criticidade,
            SUM(CASE WHEN rating IN ('Alto', 'Médio', 'Medio') THEN 1 ELSE 0 END) AS qtd_riscos_altos_medios,
            SUM(CASE WHEN impacto IN ('Crítico', 'Critico') THEN 1 ELSE 0 END) AS qtd_impactos_criticos,
            SUM(CASE WHEN upper(cloud) IN ('TRUE', 'VERDADEIRO', 'SIM') THEN 1 ELSE 0 END) AS qtd_itens_cloud,
            ({score_sql}) AS score_risco_servidor,
            CASE
                WHEN ({score_sql}) >= {LIMIAR_RISCO_ALTO} THEN 'Alto'
                WHEN ({score_sql}) >= {LIMIAR_RISCO_MEDIO} THEN 'Médio'
                ELSE 'Baixo'
            END AS classificacao_risco_servidor
        FROM {database}.{gold_table}
        GROUP BY servidor, sigla_origem, ambiente, arquitetura
    """)

    spark.sql(f"""
        CREATE OR REPLACE VIEW {database}.{VW_DETALHE_OBSOLESCENCIA_POR_SERVIDOR} AS
        SELECT
            servidor,
            tipo_chave,
            descricao_servidor,
            sigla_origem,
            ambiente,
            arquitetura,
            status_servidor,
            status_operacional,
            software_instalado,
            software_catalogado,
            versao,
            produto_tecnologico,
            criticidade,
            indice_obsolescencia,
            rating,
            impacto,
            cloud,
            detalhe_cabecalho,
            data_processamento
        FROM {database}.{gold_table}
    """)

    spark.sql(f"""
        CREATE OR REPLACE VIEW {database}.{VW_RESUMO_RISCO_POR_SERVIDOR} AS
        SELECT
            servidor,
            sigla_origem,
            COUNT(DISTINCT software_instalado) AS qtd_softwares_instalados,
            COUNT(DISTINCT produto_tecnologico) AS qtd_produtos_tecnologicos,
            AVG(indice_obsolescencia) AS media_indice_obsolescencia,
            AVG(criticidade) AS media_criticidade,
            SUM(CASE WHEN rating IN ('Alto', 'Médio', 'Medio') THEN 1 ELSE 0 END) AS qtd_riscos_altos_medios,
            SUM(CASE WHEN impacto IN ('Crítico', 'Critico') THEN 1 ELSE 0 END) AS qtd_impactos_criticos
        FROM {database}.{gold_table}
        GROUP BY servidor, sigla_origem
    """)


# ============================================================
# Execução principal
# ============================================================

def main(spark: SparkSession, job_name: str) -> None:
    logger.info("Iniciando job: %s", job_name)
    job_start = time.perf_counter()

    servidores, relacionamentos, software_instance, software_catalogo = read_silver_tables(spark, DATABASE_NAME)

    servidores_tratado = treat_servers(servidores)
    software_inst_tratado = treat_software_instances(software_instance)
    software_cat_tratado = treat_software_catalog(software_catalogo)
    relacionamentos_tratado = treat_relationships(relacionamentos)

    gold = build_gold_por_servidor(
        servidores_tratado,
        software_inst_tratado,
        software_cat_tratado,
        relacionamentos_tratado,
    )

    write_gold(spark, gold, DATABASE_NAME, GOLD_TABLE, GOLD_PATH)
    create_views_por_servidor(spark, DATABASE_NAME, GOLD_TABLE)

    logger.info("Job finalizado em %.2fs", time.perf_counter() - job_start)


args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

main(spark, args["JOB_NAME"])

job.commit()