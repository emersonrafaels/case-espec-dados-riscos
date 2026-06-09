CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_resumo_risco_por_sigla AS
SELECT
    classificacao_risco,
    COUNT(*) AS qtd_siglas,
    ROUND(AVG(score_risco_operacional),2) AS score_medio,
    SUM(qtd_produtos_tecnologicos) AS qtd_produtos_tecnologicos,
    SUM(qtd_servidores) AS qtd_servidores
FROM workspace_db_case_espec_dados_riscos.gold_risco_tecnologico_por_sigla
GROUP BY classificacao_risco;