import sys
import time
import logging
from typing import Tuple

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import DataFrame, SparkSession, Window
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

GOLD_TABLE = "case_riscos_gold_obsolescencia_por_servidor"
GOLD_PATH = f"s3://{BUCKET_NAME}/case-riscos/gold/{GOLD_TABLE}/"

VW_RANKING_RISCO_POR_SERVIDOR = "case_riscos_vw_ranking_risco_por_servidor"
VW_DETALHE_OBSOLESCENCIA_POR_SERVIDOR = "case_riscos_vw_detalhe_obsolescencia_por_servidor"
VW_RESUMO_RISCO_POR_SERVIDOR = "case_riscos_vw_resumo_risco_por_servidor"

PESO_OBSOLESCENCIA_SERVIDOR = 10.0
PESO_CRITICIDADE_SERVIDOR = 10.0
PESO_RISCO_ALTO_MEDIO = 2.0
PESO_IMPACTO_CRITICO = 3.0

LIMIAR_RISCO_ALTO = 60.0
LIMIAR_RISCO_MEDIO = 20.0

ANOMES_DEFAULT = "202604"
MIN_CHARS_MATCH_CONTEM = 4


# ============================================================
# Funções de transformação de colunas
# ============================================================

def normalize_join_key(expr):
    """
    Normaliza uma expressão Spark para chave de join:
    - cast string;
    - trim;
    - lower;
    - remove múltiplos espaços.
    """
    return F.lower(F.trim(F.regexp_replace(expr.cast("string"), r"\s+", " ")))


def to_double_from_ptbr(column: str):
    """
    Converte número em formato string PT-BR para double.
    Exemplo: '1,25' -> 1.25.
    """
    return F.regexp_replace(F.trim(F.col(column).cast("string")), ",", ".").cast(DoubleType())


def clean_empty_string(column: str):
    """
    Padroniza valores vazios, nulos, 'nan' e 'empty string' como null.
    """
    return (
        F.when(
            F.col(column).isNull()
            | (F.lower(F.trim(F.col(column).cast("string"))) == "nan")
            | (F.lower(F.trim(F.col(column).cast("string"))) == "empty string")
            | (F.trim(F.col(column).cast("string")) == ""),
            None,
        )
        .otherwise(F.trim(F.col(column).cast("string")))
    )


def contains_col(left_col, right_col):
    """
    Verifica se uma coluna contém outra coluna.
    Exige tamanho mínimo para reduzir falso positivo em matches por substring.

    F.instr(str, substr) exige que substr seja um literal Python — não uma Column.
    Usamos F.expr() com a representação SQL de cada coluna para contornar essa limitação.
    """
    left_sql = left_col._jc.toString()
    right_sql = right_col._jc.toString()
    return (
        left_col.isNotNull()
        & right_col.isNotNull()
        & (F.length(right_col) >= F.lit(MIN_CHARS_MATCH_CONTEM))
        & F.expr(f"instr({left_sql}, {right_sql}) > 0")
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
    """Limpa e padroniza servidores, criando chaves para join com software e sigla."""
    return (
        df.select(
            clean_empty_string("nome").alias("servidor"),
            clean_empty_string("arquitetura").alias("arquitetura"),
            clean_empty_string("status").alias("status_servidor"),
            clean_empty_string("status_operacional").alias("status_operacional"),
            clean_empty_string("ambiente").alias("ambiente"),
            clean_empty_string("sigla").alias("sigla_origem"),
        )
        .withColumn("servidor_join", normalize_join_key(F.col("servidor")))
        .withColumn("sigla_join", normalize_join_key(F.col("sigla_origem")))
    )


def treat_software_instances(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza instâncias de software instaladas em servidores.
    A coluna software normalmente vem no padrão: name + version.
    """
    return (
        df.select(
            clean_empty_string("installed_on").alias("servidor_instancia"),
            clean_empty_string("software").alias("software_instalado"),
            clean_empty_string("anomesdia").alias("anomesdia_software_instance"),
        )
        .withColumn("servidor_join", normalize_join_key(F.col("servidor_instancia")))
        .withColumn("software_join", normalize_join_key(F.col("software_instalado")))
    )


def treat_software_catalog(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza catálogo CMDB de pacotes de software.

    Chave principal:
        software_instance.software = cmdb_ci_spkg_sot.name + ' ' + cmdb_ci_spkg_sot.version

    Chaves auxiliares:
        name_join, version_join e model_join para matches alternativos.
    """
    return (
        df.select(
            clean_empty_string("model_id").alias("model_id"),
            clean_empty_string("name").alias("nome_software_catalogo"),
            clean_empty_string("version").alias("versao_catalogo"),
            clean_empty_string("anomesdia").alias("anomesdia_catalogo"),
        )
        .withColumn("name_join", normalize_join_key(F.col("nome_software_catalogo")))
        .withColumn("version_join", normalize_join_key(F.col("versao_catalogo")))
        .withColumn("model_join", normalize_join_key(F.col("model_id")))
        .withColumn(
            "software_catalogo_join",
            normalize_join_key(F.concat_ws(" ", F.col("nome_software_catalogo"), F.col("versao_catalogo"))),
        )
        .dropDuplicates(["software_catalogo_join", "name_join", "version_join"])
    )


def treat_relationships(df: DataFrame) -> DataFrame:
    """Limpa relacionamentos de risco tecnológico por sigla/produto."""
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
        .withColumn("sigla_join", normalize_join_key(F.col("sigla_relacionamento")))
        .withColumn("produto_join", normalize_join_key(F.col("produto_tecnologico")))
        .withColumn("versao_join", normalize_join_key(F.col("versao_relacionamento")))
        .withColumn(
            "produto_versao_join",
            normalize_join_key(F.concat_ws(" ", F.col("produto_tecnologico"), F.col("versao_relacionamento"))),
        )
    )


# ============================================================
# Construção da Gold por Servidor
# ============================================================

def add_catalog_match_priority(df: DataFrame) -> DataFrame:
    """Classifica o tipo de match entre software instalado e catálogo CMDB."""
    exact_name_version = F.col("inst.software_join") == F.col("cat.software_catalogo_join")
    contains_name_version = contains_col(F.col("inst.software_join"), F.col("cat.name_join")) & contains_col(
        F.col("inst.software_join"), F.col("cat.version_join")
    )
    contains_name = contains_col(F.col("inst.software_join"), F.col("cat.name_join"))

    return (
        df.withColumn(
            "prioridade_match_catalogo",
            F.when(exact_name_version, F.lit(1))
            .when(contains_name_version, F.lit(2))
            .when(contains_name, F.lit(3))
            .otherwise(F.lit(99)),
        )
        .withColumn(
            "tipo_match_catalogo",
            F.when(exact_name_version, F.lit("MATCH_EXATO_NAME_VERSION"))
            .when(contains_name_version, F.lit("MATCH_CONTEM_NAME_VERSION"))
            .when(contains_name, F.lit("MATCH_CONTEM_NAME"))
            .otherwise(F.lit("SEM_MATCH_CATALOGO")),
        )
    )


def add_relationship_match_priority(df: DataFrame) -> DataFrame:
    """Classifica o tipo de match com a base de obsolescência / resultado_query3."""
    exact_cat_prod_version = F.col("cat.software_catalogo_join") == F.col("rel.produto_versao_join")
    exact_cat_prod = F.col("cat.name_join") == F.col("rel.produto_join")
    exact_inst_prod_version = F.col("inst.software_join") == F.col("rel.produto_versao_join")
    exact_inst_prod = F.col("inst.software_join") == F.col("rel.produto_join")
    contains_inst_prod_version = contains_col(F.col("inst.software_join"), F.col("rel.produto_join")) & contains_col(
        F.col("inst.software_join"), F.col("rel.versao_join")
    )
    contains_inst_prod = contains_col(F.col("inst.software_join"), F.col("rel.produto_join"))

    return (
        df.withColumn(
            "prioridade_match_relacionamento",
            F.when(exact_cat_prod_version, F.lit(1))
            .when(exact_cat_prod, F.lit(2))
            .when(exact_inst_prod_version, F.lit(3))
            .when(exact_inst_prod, F.lit(4))
            .when(contains_inst_prod_version, F.lit(5))
            .when(contains_inst_prod, F.lit(6))
            .otherwise(F.lit(99)),
        )
        .withColumn(
            "tipo_match_relacionamento",
            F.when(exact_cat_prod_version, F.lit("MATCH_CAT_PRODUTO_VERSION"))
            .when(exact_cat_prod, F.lit("MATCH_CAT_PRODUTO"))
            .when(exact_inst_prod_version, F.lit("MATCH_INST_PRODUTO_VERSION"))
            .when(exact_inst_prod, F.lit("MATCH_INST_PRODUTO"))
            .when(contains_inst_prod_version, F.lit("MATCH_CONTEM_PRODUTO_VERSION"))
            .when(contains_inst_prod, F.lit("MATCH_CONTEM_PRODUTO"))
            .otherwise(F.lit("SEM_MATCH_RELACIONAMENTO")),
        )
    )


def build_gold_por_servidor(
    servidores: DataFrame,
    software_instance: DataFrame,
    software_catalogo: DataFrame,
    relacionamentos: DataFrame,
) -> DataFrame:
    """
    Constrói a Gold na granularidade Servidor → Software → Versão → Obsolescência.

    Estratégia de match em cascata:
      Catálogo CMDB:
        1. software = name + version
        2. software contém name e version
        3. software contém name

      Resultado_QUERY3:
        1. cat(name + version) = produto + versão
        2. cat(name) = produto
        3. software_instance = produto + versão
        4. software_instance = produto
        5. software_instance contém produto e versão
        6. software_instance contém produto

    Em caso de múltiplos candidatos, mantém apenas o melhor match por ranking.
    """
    base = servidores.alias("srv").join(
        software_instance.alias("inst"),
        on="servidor_join",
        how="left",
    )

    cond_catalogo = (
        (F.col("inst.software_join") == F.col("cat.software_catalogo_join"))
        | (
            contains_col(F.col("inst.software_join"), F.col("cat.name_join"))
            & contains_col(F.col("inst.software_join"), F.col("cat.version_join"))
        )
        | contains_col(F.col("inst.software_join"), F.col("cat.name_join"))
    )

    catalogo_candidatos = add_catalog_match_priority(
        base.join(software_catalogo.alias("cat"), cond_catalogo, how="left")
    )

    w_catalogo = Window.partitionBy(
        F.col("srv.servidor_join"),
        F.col("inst.software_join"),
    ).orderBy(
        F.col("prioridade_match_catalogo").asc(),
        F.length(F.col("cat.name_join")).desc_nulls_last(),
        F.length(F.col("cat.version_join")).desc_nulls_last(),
    )

    melhor_catalogo = (
        catalogo_candidatos.withColumn("rn_catalogo", F.row_number().over(w_catalogo))
        .filter(F.col("rn_catalogo") == 1)
        .drop("rn_catalogo")
    )

    cond_relacionamento = (
        (F.col("srv.sigla_join") == F.col("rel.sigla_join"))
        & (
            (F.col("cat.software_catalogo_join") == F.col("rel.produto_versao_join"))
            | (F.col("cat.name_join") == F.col("rel.produto_join"))
            | (F.col("inst.software_join") == F.col("rel.produto_versao_join"))
            | (F.col("inst.software_join") == F.col("rel.produto_join"))
            | (
                contains_col(F.col("inst.software_join"), F.col("rel.produto_join"))
                & contains_col(F.col("inst.software_join"), F.col("rel.versao_join"))
            )
            | contains_col(F.col("inst.software_join"), F.col("rel.produto_join"))
        )
    )

    relacionamento_candidatos = add_relationship_match_priority(
        melhor_catalogo.join(relacionamentos.alias("rel"), cond_relacionamento, how="left")
    )

    w_relacionamento = Window.partitionBy(
        F.col("srv.servidor_join"),
        F.col("inst.software_join"),
        F.col("cat.software_catalogo_join"),
    ).orderBy(
        F.col("prioridade_match_relacionamento").asc(),
        F.col("rel.indice_obsolescencia").desc_nulls_last(),
        F.col("rel.criticidade").desc_nulls_last(),
    )

    melhor_relacionamento = (
        relacionamento_candidatos.withColumn("rn_relacionamento", F.row_number().over(w_relacionamento))
        .filter(F.col("rn_relacionamento") == 1)
        .drop("rn_relacionamento")
    )

    return (
        melhor_relacionamento.select(
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
            F.col("tipo_match_catalogo"),
            F.col("tipo_match_relacionamento"),
            F.col("prioridade_match_catalogo"),
            F.col("prioridade_match_relacionamento"),
            F.concat_ws(
                " | ",
                F.concat(F.lit("servidor="), F.col("srv.servidor")),
                F.concat(F.lit("sigla_origem="), F.col("srv.sigla_origem")),
                F.concat(F.lit("software="), F.coalesce(F.col("inst.software_instalado"), F.lit("N/A"))),
                F.concat(F.lit("software_catalogado="), F.coalesce(F.col("cat.nome_software_catalogo"), F.lit("N/A"))),
                F.concat(F.lit("produto_tecnologico="), F.coalesce(F.col("rel.produto_tecnologico"), F.lit("N/A"))),
                F.concat(F.lit("versao="), F.coalesce(F.col("cat.versao_catalogo"), F.col("rel.versao_relacionamento"), F.lit("N/A"))),
                F.concat(F.lit("tipo_match_catalogo="), F.col("tipo_match_catalogo")),
                F.concat(F.lit("tipo_match_relacionamento="), F.col("tipo_match_relacionamento")),
            ).alias("detalhe_cabecalho"),
            F.current_timestamp().alias("data_processamento"),
            F.lit("gold").alias("camada"),
            F.lit(ANOMES_DEFAULT).alias("anomes"),
        )
        .dropDuplicates()
    )


def write_gold(spark: SparkSession, df: DataFrame, database: str, table: str, path: str) -> None:
    """Persiste DataFrame como tabela Gold no Glue Data Catalog, particionado por anomes."""
    full_name = f"{database}.{table}"
    logger.info("Gravando tabela: %s -> %s", full_name, path)

    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    (
        df.write
        .mode("overwrite")
        .format("parquet")
        .option("compression", "snappy")
        .option("path", path)
        .partitionBy("anomes")
        .saveAsTable(full_name)
    )

    logger.info("Tabela gravada: %s -> %s", full_name, path)


# ============================================================
# Views analíticas por Servidor
# ============================================================

def create_views_por_servidor(spark: SparkSession, database: str, gold_table: str) -> None:
    """Cria views analíticas por servidor, compatíveis com as visões por sigla."""
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
            tipo_match_catalogo,
            tipo_match_relacionamento,
            detalhe_cabecalho,
            data_processamento,
            anomes
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
