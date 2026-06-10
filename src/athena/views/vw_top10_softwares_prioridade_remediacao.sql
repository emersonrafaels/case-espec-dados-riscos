CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_top10_softwares_prioridade_remediacao AS

SELECT
    software_instalado,
    versao,

    COUNT(DISTINCT servidor) AS qtd_servidores_afetados,

    COUNT(*) AS qtd_ocorrencias,

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
    ) AS qtd_impactos_criticos,

    ROUND(
        (
            COUNT(DISTINCT servidor) * 5
            +
            SUM(
                CASE
                    WHEN lower(rating) IN ('alto','médio','medio')
                    THEN 1
                    ELSE 0
                END
            ) * 3
            +
            SUM(
                CASE
                    WHEN lower(impacto) LIKE '%cr%'
                    THEN 1
                    ELSE 0
                END
            ) * 4
            +
            COALESCE(AVG(indice_obsolescencia),0) * 10
            +
            COALESCE(AVG(criticidade),0) * 10
        ),
        2
    ) AS score_prioridade_remediacao

FROM workspace_db_case_espec_dados_riscos.gold_obsolescencia_por_servidor

WHERE
    indice_obsolescencia > 0
    OR lower(coalesce(rating,'')) <> 'up to date'

GROUP BY
    software_instalado,
    versao

ORDER BY
    score_prioridade_remediacao DESC

LIMIT 10;