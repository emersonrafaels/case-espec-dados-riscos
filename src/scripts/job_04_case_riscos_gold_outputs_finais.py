import sys
import time
import logging
from typing import Optional

from pyspark.context import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions


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
BUCKET        = "workspace-db-case-espec-dados-riscos"

GOLD_HUB_TABLE      = "gold_hub_analitico_servidor"
GOLD_CONTROLE_TABLE = "gold_controle_inconformidades"

GOLD_HUB_PATH      = f"s3://{BUCKET}/gold/{GOLD_HUB_TABLE}/"
GOLD_CONTROLE_PATH = f"s3://{BUCKET}/gold/{GOLD_CONTROLE_TABLE}/"

# --- Limiares do farol de inconformidade (percentual, inclusive) ---
FAROL_VERDE_MAX   = 5.0   # <= 5%  → Verde
FAROL_AMARELO_MAX = 20.0  # < 20%  → Amarelo; acima disso → Vermelho


# ============================================================
# Funções puras de transformação de colunas (expressões Spark)
# ============================================================

def normalize_text(column: str):
    """
    Retorna expressão Spark: lower(trim(col)).
    Usado para padronizar chaves de join e evitar falhas por diferença de caixa ou espaços.
    """
    return F.lower(F.trim(F.col(column)))


def to_double_from_brazilian_number(column: str):
    """
    Converte números armazenados como texto no padrão brasileiro para double.
    Troca vírgula por ponto antes do cast.
    Exemplo: '1,75' -> 1.75 | '5' -> 5.0
    """
    return (
        F.regexp_replace(F.col(column).cast("string"), ",", ".")
        .cast(DoubleType())
    )


# ============================================================
# Funções puras de leitura
# ============================================================

def table_exists(spark: SparkSession, database: str, table_name: str) -> bool:
    """
    Verifica se uma tabela existe no Glue Data Catalog.
    Evita uso de spark.catalog.tableExists() que pode não estar disponível em versões antigas do Glue.
    """
    try:
        spark.table(f"{database}.{table_name}")
        return True
    except Exception:
        return False


def read_silver_tables(
    spark: SparkSession, database: str
) -> tuple:
    """
    Lê as tabelas obrigatórias da camada Silver (servidores e riscos).
    Lança AnalysisException se qualquer tabela obrigatória não existir.
    """
    def _read(table_name: str) -> DataFrame:
        full_name = f"{database}.{table_name}"
        logger.info("Lendo tabela do catálogo: %s", full_name)
        return spark.table(full_name)

    return _read("servidores_siglas"), _read("resultado_query3")


def read_siglas_table(
    spark: SparkSession, database: str
) -> Optional[DataFrame]:
    """
    Tenta carregar a base de siglas enriquecida (Diretoria, Comunidade, Criticidade Tier).
    Procura por 'resultado_query1' primeiro, depois 'base_siglas_tiers'.
    Retorna None se nenhuma das tabelas existir no catálogo.
    """
    for candidate in ("resultado_query1", "base_siglas_tiers"):
        if table_exists(spark, database, candidate):
            full_name = f"{database}.{candidate}"
            logger.info("Base de siglas encontrada: %s", full_name)
            return spark.table(full_name)

    logger.warning(
        "Nenhuma base de siglas encontrada (%s.resultado_query1 / %s.base_siglas_tiers). "
        "Diretoria/Comunidade/Criticidade serão preenchidos como 'Não informado'.",
        database, database
    )
    return None


# ============================================================
# Funções puras de tratamento
# ============================================================

def treat_servers(df: DataFrame) -> DataFrame:
    """
    Padroniza a tabela de servidores:
    - cria chave 'servidor' (minusc + trim) para join;
    - mantém 'descricao_servidor' com o nome original;
    - filtra linhas sem servidor ou sigla (dados obrigatórios para o join).
    """
    return (
        df
        .select(
            F.lower(F.trim(F.col("nome"))).alias("servidor"),
            F.col("nome").alias("descricao_servidor"),
            F.lower(F.trim(F.col("sigla"))).alias("sigla"),
            F.col("arquitetura"),
            F.col("status"),
            F.col("status_operacional"),
            F.col("ambiente"),
        )
        # Filtra linhas sem chave de join — não é possível associar ao risco sem servidor e sigla
        .filter(F.col("servidor").isNotNull())
        .filter(F.col("sigla").isNotNull())
    )


def treat_risks(df: DataFrame) -> DataFrame:
    """
    Padroniza a tabela de riscos tecnológicos:
    - normaliza sigla como chave de join;
    - converte índices numéricos de string PT-BR para double;
    - mantém colunas originais como string para uso no campo de detalhe.
    """
    return (
        df
        .select(
            F.lower(F.trim(F.col("sigla_ss"))).alias("sigla"),
            F.col("produto"),
            F.col("versao"),
            F.col("status").alias("status_tecnologia"),
            # Mantido como string para exibição no Detalhe_Cabecalho
            F.col("criticidade").alias("criticidade_num"),
            # Cast para double necessário para cálculos numéricos
            to_double_from_brazilian_number("criticidade").alias("criticidade_num_double"),
            to_double_from_brazilian_number("indice_obsolescencia").alias("indice_obsolescencia_num"),
            F.col("indice_obsolescencia"),
            F.col("rating"),
            F.col("impacto"),
            F.col("cloud"),
        )
        .filter(F.col("sigla").isNotNull())
    )


def treat_siglas(
    siglas_df: Optional[DataFrame],
    servidores_std: DataFrame,
) -> DataFrame:
    """
    Prepara a tabela de siglas com Diretoria, Comunidade e Criticidade Tier.
    Se siglas_df for None, cria um DataFrame de fallback com 'Não informado'
    a partir das siglas já presentes em servidores.

    Usa coalesce() para aceitar tanto colunas CamelCase quanto snake_case,
    tornando a função tolerante a variações de naming entre fontes.
    """
    if siglas_df is not None:
        cols = siglas_df.columns

        # Cada campo aceita tanto 'NomeCamelCase' quanto 'nome_snake_case'.
        # coalesce() retorna o primeiro valor não-nulo entre as opções.
        return (
            siglas_df
            .select(
                F.lower(F.trim(F.coalesce(
                    F.col("Sigla_SS")  if "Sigla_SS"  in cols else F.lit(None),
                    F.col("sigla_ss")  if "sigla_ss"  in cols else F.lit(None),
                    F.col("Chave")     if "Chave"     in cols else F.lit(None),
                    F.col("chave")     if "chave"     in cols else F.lit(None),
                ))).alias("sigla"),
                F.coalesce(
                    F.col("Diretoria")  if "Diretoria"  in cols else F.lit(None),
                    F.col("diretoria")  if "diretoria"  in cols else F.lit(None),
                ).alias("diretoria"),
                F.coalesce(
                    F.col("Comunidade") if "Comunidade" in cols else F.lit(None),
                    F.col("comunidade") if "comunidade" in cols else F.lit(None),
                ).alias("comunidade"),
                F.coalesce(
                    F.col("Criticidade_Final") if "Criticidade_Final" in cols else F.lit(None),
                    F.col("criticidade_final") if "criticidade_final" in cols else F.lit(None),
                ).alias("criticidade_tier"),
            )
            .filter(F.col("sigla").isNotNull())
            # Remove siglas duplicadas — join de lookup deve ser 1:N, não N:N
            .dropDuplicates(["sigla"])
        )

    # Fallback: constrói o DataFrame de siglas a partir dos próprios servidores
    logger.info("Usando fallback para base de siglas (dados não disponíveis no catálogo).")
    return (
        servidores_std
        .select("sigla")
        .dropDuplicates()
        .withColumn("diretoria",        F.lit("Não informado"))
        .withColumn("comunidade",       F.lit("Não informado"))
        .withColumn("criticidade_tier", F.lit("Não informado"))
    )


# ============================================================
# Funções puras de construção dos outputs Gold
# ============================================================

def build_hub_analitico(
    servidores_std: DataFrame,
    riscos_std: DataFrame,
    siglas_std: DataFrame,
) -> DataFrame:
    """
    Constrói o HUB analítico de obsolescência tecnológica por servidor.

    Join: inner entre servidores e riscos (pela sigla) para garantir que
    apenas servidores com risco mapeado entrem no hub.
    Left join com siglas para enriquecer com Diretoria/Comunidade/Criticidade.

    Conformidade: qualquer servidor com índice de obsolescência > 0
    OU com rating informado é classificado como 'Não esta em conformidade'.
    """
    return (
        servidores_std.alias("srv")
        # Inner join: só servidores que têm risco tecnológico mapeado
        .join(riscos_std.alias("risco"), on="sigla", how="inner")
        # Left join: sigla pode não existir na base de enriquecimento
        .join(siglas_std.alias("sig"),   on="sigla", how="left")

        # Conformidade: obsolescência > 0 OU qualquer rating implica inconformidade
        .withColumn(
            "Conformidade",
            F.when(
                (F.col("risco.indice_obsolescencia_num") > 0)
                | F.lower(F.col("risco.rating")).isin("alto", "médio", "medio", "baixo"),
                F.lit("Não esta em conformidade")
            ).otherwise(F.lit("Em conformidade"))
        )

        # Detalhe_Cabecalho: concatena todos os atributos-chave para auditoria linha a linha
        .withColumn(
            "Detalhe_Cabecalho",
            F.concat_ws(
                " | ",
                F.concat(F.lit("sigla_origem="),         F.col("sigla")),
                F.concat(F.lit("produto="),              F.coalesce(F.col("risco.produto"),              F.lit(""))),
                F.concat(F.lit("versao="),               F.coalesce(F.col("risco.versao"),               F.lit(""))),
                F.concat(F.lit("status="),               F.coalesce(F.col("risco.status_tecnologia"),    F.lit(""))),
                F.concat(F.lit("criticidade="),          F.coalesce(F.col("risco.criticidade_num"),      F.lit(""))),
                F.concat(F.lit("indice_obsolescencia="), F.coalesce(F.col("risco.indice_obsolescencia"), F.lit(""))),
                F.concat(F.lit("rating="),               F.coalesce(F.col("risco.rating"),               F.lit(""))),
                F.concat(F.lit("impacto="),              F.coalesce(F.col("risco.impacto"),              F.lit(""))),
                F.concat(F.lit("cloud="),                F.coalesce(F.col("risco.cloud"),                F.lit(""))),
            )
        )

        .select(
            F.lit("Obsolescência tecnológica por servidor").alias("INDICADOR"),
            F.lit("Controle de softwares obsoletos por servidor").alias("Nome_Catch"),
            F.col("srv.servidor").alias("Chave"),
            F.lit("Servidor").alias("Tipo_chave"),
            F.col("srv.descricao_servidor").alias("Descricao_chave"),
            F.coalesce(F.col("sig.criticidade_tier"), F.lit("Não informado")).alias("Criticidade"),
            F.coalesce(F.col("sig.diretoria"),         F.lit("Não informado")).alias("Diretoria"),
            F.coalesce(F.col("sig.comunidade"),        F.lit("Não informado")).alias("Comunidade"),
            F.col("Conformidade"),
            # Nome_Cabecalho descreve o schema do Detalhe_Cabecalho para o consumidor do dado
            F.lit("sigla_origem | produto | versao | status | criticidade | indice_obsolescencia | rating | impacto | cloud").alias("Nome_Cabecalho"),
            F.col("Detalhe_Cabecalho"),
        )
    )


def build_controle_inconformidades(hub_analitico: DataFrame) -> DataFrame:
    """
    Agrega o HUB analítico para calcular percentual de inconformidade e farol.

    Farol:
      Verde    : percentual <= FAROL_VERDE_MAX   (5%)
      Amarelo  : percentual <  FAROL_AMARELO_MAX (20%)
      Vermelho : demais casos

    Os limiares são configuráveis pelas constantes FAROL_* no bloco de configuração.
    """
    return (
        hub_analitico
        .agg(
            F.count("*").alias("total_registros"),
            # Conta apenas as linhas inconformes usando SUM com condição
            F.sum(
                F.when(F.col("Conformidade") == "Não esta em conformidade", 1).otherwise(0)
            ).alias("total_inconformes"),
        )
        # Percentual com 2 casas decimais para facilitar leitura
        .withColumn(
            "percentual_inconformidade",
            F.round((F.col("total_inconformes") / F.col("total_registros")) * 100, 2),
        )
        # Classificação do farol baseada nos limiares configurados
        .withColumn(
            "farol",
            F.when(F.col("percentual_inconformidade") <= FAROL_VERDE_MAX,   F.lit("Verde"))
             .when(F.col("percentual_inconformidade") <  FAROL_AMARELO_MAX, F.lit("Amarelo"))
             .otherwise(F.lit("Vermelho"))
        )
        # Documenta o critério usado no próprio dado para rastreabilidade
        .withColumn(
            "criterio_farol",
            F.lit(f"Verde <= {FAROL_VERDE_MAX}%; Amarelo > {FAROL_VERDE_MAX}% e < {FAROL_AMARELO_MAX}%; Vermelho >= {FAROL_AMARELO_MAX}%"),
        )
        .withColumn("data_processamento", F.current_timestamp())
    )


def write_gold_table(
    spark: SparkSession,
    df: DataFrame,
    database: str,
    table_name: str,
    path: str,
) -> None:
    """
    Persiste um DataFrame Gold em Parquet comprimido (Snappy) no S3
    e registra/atualiza a entrada no Glue Data Catalog via saveAsTable.
    DROP TABLE IF EXISTS antes do write para garantir consistência do schema
    em reprocessamentos (evita conflito de schema evolution com saveAsTable).
    """
    full_name = f"{database}.{table_name}"
    logger.info("Removendo tabela existente antes da recriação: %s", full_name)
    spark.sql(f"DROP TABLE IF EXISTS {full_name}")

    (
        df.write
        .mode("overwrite")                   # Sobrescreve completamente — garante idempotência
        .format("parquet")
        .option("compression", "snappy")     # Snappy: melhor equilíbrio velocidade/tamanho para Athena
        .option("path", path)                # Grava fisicamente no prefixo S3 definido
        .saveAsTable(full_name)              # Registra/atualiza entrada no Glue Data Catalog
    )
    logger.info("Tabela Gold gravada: %s -> %s", full_name, path)


def main(spark: SparkSession, job_name: str) -> None:
    """
    Orquestra o pipeline Silver → Gold (outputs finais) como composição de funções puras.

    Fluxo:
        read_silver_tables + read_siglas_table
            → treat_servers                  ┐
            → treat_risks                     |
            → treat_siglas (ou fallback)      ├→ build_hub_analitico
                                              |       → build_controle_inconformidades
                                              ┘→ write_gold_table (x2)
    """
    logger.info("Iniciando job: %s", job_name)
    job_start = time.perf_counter()

    # --- Fase 1: Leitura ---
    logger.info("=== Fase 1: Leitura das tabelas Silver ===")
    try:
        servidores, riscos = read_silver_tables(spark, DATABASE_NAME)
        siglas_raw         = read_siglas_table(spark, DATABASE_NAME)
    except Exception as exc:
        logger.error("Falha na leitura das tabelas Silver: %s", exc, exc_info=True)
        raise

    # --- Fase 2: Tratamento ---
    logger.info("=== Fase 2: Tratamento das bases ===")
    _t = time.perf_counter()

    servidores_std = treat_servers(servidores)
    riscos_std     = treat_risks(riscos)
    # treat_siglas trata tanto o caso com dados quanto o fallback sem dados
    siglas_std     = treat_siglas(siglas_raw, servidores_std)

    logger.info("Tratamento definido em %.2fs.", time.perf_counter() - _t)

    # --- Fase 3: Construção dos outputs Gold ---
    logger.info("=== Fase 3: Construção dos outputs Gold ===")
    _t = time.perf_counter()

    hub_analitico            = build_hub_analitico(servidores_std, riscos_std, siglas_std)
    # O controle é derivado do hub — depende da fase anterior
    controle_inconformidades = build_controle_inconformidades(hub_analitico)

    logger.info("Outputs Gold definidos em %.2fs.", time.perf_counter() - _t)

    # --- Fase 4: Escrita ---
    logger.info("=== Fase 4: Escrita na camada Gold ===")
    _t = time.perf_counter()

    try:
        write_gold_table(spark, hub_analitico,            DATABASE_NAME, GOLD_HUB_TABLE,      GOLD_HUB_PATH)
        write_gold_table(spark, controle_inconformidades, DATABASE_NAME, GOLD_CONTROLE_TABLE, GOLD_CONTROLE_PATH)
    except Exception as exc:
        logger.error("Falha ao gravar tabelas Gold: %s", exc, exc_info=True)
        raise

    logger.info("Tabelas Gold gravadas em %.2fs.", time.perf_counter() - _t)
    logger.info(
        "Tabelas Gold finais criadas com sucesso: %s.%s | %s.%s",
        DATABASE_NAME, GOLD_HUB_TABLE, DATABASE_NAME, GOLD_CONTROLE_TABLE
    )
    logger.info("Job finalizado em %.2fs.", time.perf_counter() - job_start)


# ============================================================
# Inicialização do Glue Job e execução
# ============================================================

# Lê parâmetros injetados pelo Glue no momento da execução do job
args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(args["JOB_NAME"], args)

main(spark, args["JOB_NAME"])

job.commit()