CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_resumo_obsolescencia_por_servidor AS
SELECT
    sigla_origem,

    COUNT(DISTINCT servidor) AS qtd_servidores,

    COUNT(DISTINCT software_instalado) AS qtd_softwares,

    ROUND(AVG(indice_obsolescencia),2) AS media_indice_obsolescencia,

    ROUND(AVG(criticidade),2) AS media_criticidade,

    SUM(
        CASE
            WHEN lower(rating) IN ('alto','médio','medio')
            THEN 1
            ELSE 0
        END
    ) AS qtd_riscos_altos_medios,

    SUM(
        CASE
            WHEN lower(impacto) LIKE '%cr%'
            THEN 1
            ELSE 0
        END
    ) AS qtd_impactos_criticos

FROM workspace_db_case_espec_dados_riscos.gold_obsolescencia_por_servidor

GROUP BY sigla_origem;