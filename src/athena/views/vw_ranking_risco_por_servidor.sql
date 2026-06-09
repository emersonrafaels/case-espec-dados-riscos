CREATE OR REPLACE VIEW workspace_db.case_riscos_vw_ranking_risco_por_servidor AS
SELECT
    servidor,
    sigla_origem,
    ambiente,
    arquitetura,

    COUNT(DISTINCT software_instalado) AS qtd_softwares,

    COUNT(DISTINCT produto_tecnologico) AS qtd_produtos,

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

    (
        COALESCE(AVG(indice_obsolescencia),0) * 10
        +
        COALESCE(AVG(criticidade),0) * 10
        +
        COALESCE(
            SUM(
                CASE
                    WHEN lower(rating) IN ('alto','médio','medio')
                    THEN 1
                    ELSE 0
                END
            ),
            0
        ) * 2
        +
        COALESCE(
            SUM(
                CASE
                    WHEN lower(impacto) LIKE '%cr%'
                    THEN 1
                    ELSE 0
                END
            ),
            0
        ) * 3
    ) AS score_risco_servidor

FROM workspace_db.case_riscos_gold_obsolescencia_por_servidor

GROUP BY
    servidor,
    sigla_origem,
    ambiente,
    arquitetura;