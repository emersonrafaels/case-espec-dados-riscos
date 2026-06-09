CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_top_10_siglas_criticas AS
SELECT
    sigla,
    score_risco_operacional,
    classificacao_risco,
    qtd_produtos_tecnologicos,
    qtd_riscos_altos_medios,
    qtd_impactos_criticos,
    media_indice_obsolescencia,
    qtd_servidores
FROM workspace_db_case_espec_dados_riscos.gold_risco_tecnologico_por_sigla
ORDER BY score_risco_operacional DESC
LIMIT 10;