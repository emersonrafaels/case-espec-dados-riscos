CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_resumo_risco AS
SELECT
  classificacao_risco,
  COUNT(*) AS qtd_siglas,
  AVG(score_risco_operacional) AS score_medio,
  SUM(qtd_produtos_tecnologicos) AS qtd_produtos,
  SUM(qtd_servidores) AS qtd_servidores
FROM workspace_db_case_espec_dados_riscos.gold_risco_tecnologico_por_sigla
GROUP BY classificacao_risco;