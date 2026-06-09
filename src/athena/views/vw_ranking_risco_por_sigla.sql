CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_ranking_risco_por_sigla AS
SELECT
    sigla,
    qtd_produtos_tecnologicos,
    qtd_relacionamentos_tecnologia,
    media_indice_obsolescencia,
    media_criticidade,
    qtd_riscos_altos_medios,
    qtd_impactos_criticos,
    qtd_itens_cloud,
    qtd_servidores,
    qtd_tipos_arquitetura,
    qtd_servidores_producao,
    qtd_servidores_operacionais,
    media_softwares_por_servidor,
    score_risco_operacional,
    classificacao_risco
FROM workspace_db_case_espec_dados_riscos.gold_risco_tecnologico_por_sigla
ORDER BY score_risco_operacional DESC;