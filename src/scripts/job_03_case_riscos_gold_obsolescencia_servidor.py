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

DATABASE_NAME = "workspace_db_case_espec_dados_riscos"
BUCKET_NAME   = "workspace-db-case-espec-dados-riscos"

GOLD_TABLE = "gold_obsolescencia_servidor"
GOLD_PATH  = f"s3://{BUCKET_NAME}/gold/{GOLD_TABLE}/"

# --- Nomes das views analíticas criadas sobre a Gold ---
VW_RANKING_SERVIDORES    = "vw_ranking_servidores_criticos"
VW_DETALHE_SERVIDOR      = "vw_detalhe_obsolescencia_servidor"
VW_RESUMO_SIGLA_SERVIDOR = "vw_resumo_sigla_servidor"

# --- Pesos do score de risco por servidor (altere aqui para recalibrar o modelo) ---
PESO_OBSOLESCENCIA_SERVIDOR = 10.0
PESO_CRITICIDADE_SERVIDOR   = 10.0
PESO_RISCO_ALTO_MEDIO       = 2.0
PESO_IMPACTO_CRITICO        = 3.0

# --- Limiares de classificação do score de risco (inclusive) ---
LIMIAR_RISCO_ALTO  = 60.0
LIMIAR_RISCO_MEDIO = 20.0


# ============================================================
# Funções puras de transformação de colunas (expressões Spark)
# ============================================================

def normalize_text(column: str):
    """
    Retorna expressão Spark: lower(trim(col)).
    Usado para padronizar chaves de join e evitar falhas por diferença de caixa ou espaços.
    """
    return F.lower(F.trim(F.col(column)))


def to_double_from_ptbr(column: str):
    """
    Converte número em formato string PT-BR para double.
    Troca vírgula por ponto antes do cast para suportar valores como '1,25'.
    Exemplo: '1,25' -> 1.25 | '3.40' -> 3.40
    """
    return (
        F.regexp_replace(F.trim(F.col(column)), ",", ".")
        .cast(DoubleType())
    )


def clean_empty_string(column: str):
    """
    Padroniza como null os seguintes valores: None, 'nan', 'empty string' e string em branco.
    Garante consistência no tratamento de ausência de dados antes dos joins.
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
# Funções puras de leitura e tratamento
# ============================================================

def read_silver_tables(
    spark: SparkSession, database: str
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Lê as quatro tabelas da camada Silver do Glue Data Catalog.
    Retorna tupla: (servidores, relacionamentos, software_instance, software_catalogo).
    Lança AnalysisException se qualquer tabela não existir no catálogo.
    """
    def _read(table_name: str) -> DataFrame:
        full_name = f"{database}.{table_name}"
        logger.info("Lendo tabela do catálogo: %s", full_name)
        return spark.table(full_name)

    return (
        _read("servidores_siglas"),
        _read("resultado_query3"),
        _read("cmdb_software_instance_sot"),
        _read("cmdb_ci_spkg_sot"),
    )


def treat_servers(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza a tabela de servidores.
    Seleciona colunas relevantes, aplica clean_empty_string em todos os campos
    e cria chaves de join normalizadas (servidor_join, sigla_join).
    """
    return (
        df
        .select(
            clean_empty_string("nome").alias("servidor"),
            clean_empty_string("arquitetura").alias("arquitetura"),
            clean_empty_string("status").alias("status_servidor"),
            clean_empty_string("status_operacional").alias("status_operacional"),
            clean_empty_string("ambiente").alias("ambiente"),
            clean_empty_string("sigla").alias("sigla_origem"),
        )
        # Chave para join com software_instance (via "installed_on")
        .withColumn("servidor_join", F.lower(F.trim(F.col("servidor"))))
        # Chave para join com relacionamentos (via "sigla_ss")
        .withColumn("sigla_join",    F.lower(F.trim(F.col("sigla_origem"))))
    )


def treat_software_instances(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza a tabela de instâncias de software instalado.
    Cria servidor_join (para join com servidores) e
    software_join (para join com catálogo de softwares).
    """
    return (
        df
        .select(
            clean_empty_string("installed_on").alias("servidor_instancia"),
            clean_empty_string("software").alias("software_instalado"),
            clean_empty_string("anomesdia").alias("anomesdia_software_instance"),
        )
        .withColumn("servidor_join", F.lower(F.trim(F.col("servidor_instancia"))))
        .withColumn("software_join", F.lower(F.trim(F.col("software_instalado"))))
    )


def treat_software_catalog(df: DataFrame) -> DataFrame:
    """
    Limpa e padroniza o catálogo de pacotes de software (CMDB).
    Cria model_join e name_join para cruzamento com instâncias e relacionamentos.
    """
    return (
        df
        .select(
            clean_empty_string("model_id").alias("model_id"),
            clean_empty_string("name").alias("nome_software_catalogo"),
            clean_empty_string("version").alias("versao_catalogo"),
            clean_empty_string("anomesdia").alias("anomesdia_catalogo"),
        )
        .withColumn("model_join", F.lower(F.trim(F.col("model_id"))))
        .withColumn("name_join",  F.lower(F.trim(F.col("nome_software_catalogo"))))
    )


def treat_relationships(df: DataFrame) -> DataFrame:
    """
    Limpa a tabela de relacionamentos tecnológicos (riscos/produtos por sigla).
    Converte índices numéricos de string PT-BR para double e cria chaves de join.
    """
    return (
        df
        .select(
            clean_empty_string("sigla_ss").alias("sigla_relacionamento"),
            clean_empty_string("produto").alias("produto_tecnologico"),
            clean_empty_string("versao").alias("versao_relacionamento"),
            clean_empty_string("status").alias("status_relacionamento"),
            # Converte "0,75" → 0.75 antes do cast — padrão de número brasileiro
            to_double_from_ptbr("criticidade").alias("criticidade_num"),
            to_double_from_ptbr("indice_obsolescencia").alias("indice_obsolescencia_num"),
            clean_empty_string("rating").alias("rating"),
            clean_empty_string("impacto").alias("impacto"),
            clean_empty_string("cloud").alias("cloud"),
        )
        .withColumn("sigla_join",   F.lower(F.trim(F.col("sigla_relacionamento"))))
        .withColumn("produto_join", F.lower(F.trim(F.col("produto_tecnologico"))))
    )


def build_gold(
    servidores: DataFrame,
    software_instance: DataFrame,
    software_catalogo: DataFrame,
    relacionamentos: DataFrame,
) -> DataFrame:
    """
    Constrói a tabela Gold de obsolescência por servidor.
    Granularidade: Servidor → Software → Versão → Obsolescência.

    Estratégia de join:
      1. Servidor ←left→ instâncias de software  (via servidor_join)
      2. Instância ←left→ catálogo de software   (via name_join)
      3. (Servidor + Software) ←left→ relacionamentos de risco (via sigla + produto)

    Left joins garantem que servidores sem softwares ou sem riscos mapeados
    sejam mantidos na tabela Gold.
    dropDuplicates() remove linhas redundantes geradas pelo join múltiplo.
    """
    return (
        servidores.alias("srv")

        # Join 1: associa cada servidor às instâncias de software instaladas
        .join(
            software_instance.alias("inst"),
            on="servidor_join",
            how="left",
        )

        # Join 2: enriquece cada instância com metadados do catálogo CMDB
        .join(
            software_catalogo.alias("cat"),
            F.col("inst.software_join") == F.col("cat.name_join"),
            how="left",
        )

        # Join 3: associa o par (sigla, produto) à tabela de riscos.
        # Condição OR para aceitar match pelo nome do catálogo OU pelo nome da instância.
        .join(
            relacionamentos.alias("rel"),
            (F.col("srv.sigla_join") == F.col("rel.sigla_join"))
            & (
                (F.col("cat.name_join")        == F.col("rel.produto_join"))
                | (F.col("inst.software_join") == F.col("rel.produto_join"))
            ),
            how="left",
        )

        .select(
            F.col("srv.servidor").alias("chave"),
            F.lit("Servidor").alias("tipo_chave"),
            F.col("srv.servidor").alias("descricao_chave"),

            F.col("srv.sigla_origem"),
            F.col("srv.arquitetura"),
            F.col("srv.status_servidor"),
            F.col("srv.status_operacional"),
            F.col("srv.ambiente"),

            F.col("inst.software_instalado"),
            F.col("cat.nome_software_catalogo").alias("software_catalogado"),

            # Versão: prefere dado do catálogo; usa relacionamento como fallback
            F.coalesce(
                F.col("cat.versao_catalogo"),
                F.col("rel.versao_relacionamento"),
            ).alias("versao"),

            F.col("rel.produto_tecnologico"),
            F.col("rel.status_relacionamento"),
            F.col("rel.criticidade_num").alias("criticidade"),
            F.col("rel.indice_obsolescencia_num").alias("indice_obsolescencia"),
            F.col("rel.rating"),
            F.col("rel.impacto"),
            F.col("rel.cloud"),

            # Campo de rastreabilidade: descreve os atributos-chave da linha para auditoria
            F.concat_ws(
                " | ",
                F.concat(F.lit("servidor="),           F.col("srv.servidor")),
                F.concat(F.lit("sigla_origem="),        F.col("srv.sigla_origem")),
                F.concat(F.lit("software="),            F.coalesce(F.col("inst.software_instalado"), F.lit("N/A"))),
                F.concat(F.lit("produto_tecnologico="), F.coalesce(F.col("rel.produto_tecnologico"), F.lit("N/A"))),
                F.concat(F.lit("versao="),              F.coalesce(F.col("cat.versao_catalogo"), F.col("rel.versao_relacionamento"), F.lit("N/A"))),
            ).alias("detalhe_cabecalho"),

            F.current_timestamp().alias("data_processamento"),
            F.lit("gold").alias("camada"),
        )
        # Remove duplicatas que podem surgir do join múltiplo
        .dropDuplicates()
    )


def write_gold(df: DataFrame, database: str, table: str, path: str) -> None:
    """
    Persiste o DataFrame Gold em Parquet comprimido (Snappy) no S3
    e registra/atualiza a entrada no Glue Data Catalog via saveAsTable.
    Operação idempotente: mode("overwrite") garante reprocessamento seguro.
    """
    (
        df.write
        .mode("overwrite")                   # Sobrescreve completamente — garante idempotência
        .format("parquet")
        .option("compression", "snappy")     # Snappy: melhor equilíbrio velocidade/tamanho para Athena
        .option("path", path)                # Grava fisicamente no prefixo S3 definido
        .saveAsTable(f"{database}.{table}")  # Registra/atualiza entrada no Glue Data Catalog
    )


def create_views(
    spark: SparkSession,
    database: str,
    gold_table: str,
    vw_ranking: str,
    vw_detalhe: str,
    vw_resumo: str,
) -> None:
    """
    Cria ou substitui as três views analíticas sobre a tabela Gold.
      - vw_ranking : ranking de servidores por score de risco (GROUP BY servidor)
      - vw_detalhe : visão linha a linha de todos os atributos
      - vw_resumo  : resumo agregado por sigla

    O score SQL é construído a partir das constantes PESO_* e LIMIAR_*
    para manter consistência com o modelo definido no bloco de configuração.
    """
    # Expressão do score reutilizada no SELECT e no CASE da view de ranking.
    # Definida como string para evitar triplicação da fórmula dentro do SQL.
    score_sql = f"""
        COALESCE(AVG(indice_obsolescencia), 0) * {PESO_OBSOLESCENCIA_SERVIDOR}
        + COALESCE(AVG(criticidade), 0) * {PESO_CRITICIDADE_SERVIDOR}
        + COALESCE(SUM(CASE WHEN rating  IN ('Alto', 'M\u00e9dio', 'Medio')    THEN 1 ELSE 0 END), 0) * {PESO_RISCO_ALTO_MEDIO}
        + COALESCE(SUM(CASE WHEN impacto IN ('Cr\u00edtico', 'Critico') THEN 1 ELSE 0 END), 0) * {PESO_IMPACTO_CRITICO}
    """

    logger.info("Criando view: %s.%s", database, vw_ranking)
    spark.sql(f"""
        CREATE OR REPLACE VIEW {database}.{vw_ranking} AS
        SELECT
            chave                                    AS servidor,
            sigla_origem,
            ambiente,
            arquitetura,
            COUNT(DISTINCT software_instalado)       AS qtd_softwares_instalados,
            COUNT(DISTINCT produto_tecnologico)      AS qtd_produtos_tecnologicos,
            AVG(indice_obsolescencia)                AS media_indice_obsolescencia,
            AVG(criticidade)                         AS media_criticidade,
            SUM(CASE WHEN rating  IN ('Alto', 'M\u00e9dio', 'Medio')    THEN 1 ELSE 0 END) AS qtd_riscos_altos_medios,
            SUM(CASE WHEN impacto IN ('Cr\u00edtico', 'Critico') THEN 1 ELSE 0 END)        AS qtd_impactos_criticos,
            SUM(CASE WHEN upper(cloud) IN ('TRUE', 'VERDADEIRO', 'SIM') THEN 1 ELSE 0 END) AS qtd_itens_cloud,
            ({score_sql})                            AS score_risco_servidor,
            CASE
                WHEN ({score_sql}) >= {LIMIAR_RISCO_ALTO}  THEN 'Alto'
                WHEN ({score_sql}) >= {LIMIAR_RISCO_MEDIO} THEN 'M\u00e9dio'
                ELSE 'Baixo'
            END                                      AS classificacao_risco_servidor
        FROM {database}.{gold_table}
        GROUP BY chave, sigla_origem, ambiente, arquitetura
    """)

    logger.info("Criando view: %s.%s", database, vw_detalhe)
    spark.sql(f"""
        CREATE OR REPLACE VIEW {database}.{vw_detalhe} AS
        SELECT
            chave AS servidor,
            tipo_chave,
            descricao_chave,
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

    logger.info("Criando view: %s.%s", database, vw_resumo)
    spark.sql(f"""
        CREATE OR REPLACE VIEW {database}.{vw_resumo} AS
        SELECT
            sigla_origem,
            COUNT(DISTINCT chave)               AS qtd_servidores,
            COUNT(DISTINCT software_instalado)  AS qtd_softwares_instalados,
            AVG(indice_obsolescencia)           AS media_indice_obsolescencia,
            AVG(criticidade)                    AS media_criticidade,
            SUM(CASE WHEN rating  IN ('Alto', 'M\u00e9dio', 'Medio')    THEN 1 ELSE 0 END) AS qtd_riscos_altos_medios,
            SUM(CASE WHEN impacto IN ('Cr\u00edtico', 'Critico') THEN 1 ELSE 0 END)        AS qtd_impactos_criticos
        FROM {database}.{gold_table}
        GROUP BY sigla_origem
    """)


def main(spark: SparkSession, job_name: str) -> None:
    """
    Orquestra o pipeline Silver → Gold como composição de funções puras.

    Fluxo:
        read_silver_tables
            → treat_servers           ┐
            → treat_software_instances├─→ build_gold → write_gold → create_views
            → treat_software_catalog  |
            → treat_relationships     ┘
    """
    logger.info("Iniciando job: %s", job_name)
    job_start = time.perf_counter()

    # --- Fase 1: Leitura ---
    logger.info("=== Fase 1: Leitura das tabelas Silver ===")
    try:
        servidores, relacionamentos, software_instance, software_catalogo = \
            read_silver_tables(spark, DATABASE_NAME)
    except Exception as exc:
        logger.error("Falha na leitura das tabelas Silver: %s", exc, exc_info=True)
        raise

    # --- Fase 2: Tratamento (pipeline funcional — DataFrame in, DataFrame out) ---
    logger.info("=== Fase 2: Tratamento das bases ===")
    _t = time.perf_counter()

    servidores_tratado      = treat_servers(servidores)
    software_inst_tratado   = treat_software_instances(software_instance)
    software_cat_tratado    = treat_software_catalog(software_catalogo)
    relacionamentos_tratado = treat_relationships(relacionamentos)

    logger.info("Tratamento definido em %.2fs.", time.perf_counter() - _t)

    # --- Fase 3: Join e construção da Gold ---
    logger.info("=== Fase 3: Construção da tabela Gold ===")
    _t = time.perf_counter()

    gold = build_gold(
        servidores_tratado,
        software_inst_tratado,
        software_cat_tratado,
        relacionamentos_tratado,
    )

    logger.info("Tabela Gold definida em %.2fs.", time.perf_counter() - _t)

    # --- Fase 4: Escrita ---
    logger.info("=== Fase 4: Escrita na camada Gold ===")
    _t = time.perf_counter()

    try:
        write_gold(gold, DATABASE_NAME, GOLD_TABLE, GOLD_PATH)
    except Exception as exc:
        logger.error("Falha ao gravar tabela Gold '%s': %s", GOLD_TABLE, exc, exc_info=True)
        raise

    logger.info(
        "Tabela Gold '%s' gravada em %.2fs. Caminho: %s",
        GOLD_TABLE, time.perf_counter() - _t, GOLD_PATH
    )

    # --- Fase 5: Views analíticas ---
    logger.info("=== Fase 5: Criação das views analíticas ===")
    _t = time.perf_counter()

    try:
        create_views(
            spark, DATABASE_NAME, GOLD_TABLE,
            VW_RANKING_SERVIDORES,
            VW_DETALHE_SERVIDOR,
            VW_RESUMO_SIGLA_SERVIDOR,
        )
    except Exception as exc:
        logger.error("Falha ao criar views analíticas: %s", exc, exc_info=True)
        raise

    logger.info("Views criadas em %.2fs.", time.perf_counter() - _t)
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