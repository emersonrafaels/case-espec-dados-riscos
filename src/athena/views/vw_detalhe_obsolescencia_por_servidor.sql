CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_detalhe_obsolescencia_por_servidor AS
SELECT
    servidor,
    sigla_origem,
    ambiente,
    arquitetura,
    software_instalado,
    software_catalogado,
    versao,
    produto_tecnologico,
    criticidade,
    indice_obsolescencia,
    rating,
    impacto,
    cloud,
    detalhe_cabecalho
FROM workspace_db_case_espec_dados_riscos.gold_obsolescencia_por_servidor;