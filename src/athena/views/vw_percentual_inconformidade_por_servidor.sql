CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_percentual_inconformidade_por_servidor AS

SELECT
    servidor,
    sigla_origem,

    COUNT(*) AS qtd_total_softwares,

    SUM(
        CASE
            WHEN indice_obsolescencia > 0
              OR lower(rating) IN ('alto','médio','medio','baixo')
            THEN 1
            ELSE 0
        END
    ) AS qtd_inconformidades,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN indice_obsolescencia > 0
                  OR lower(rating) IN ('alto','médio','medio','baixo')
                THEN 1
                ELSE 0
            END
        )
        / COUNT(*)
    ,2) AS percentual_inconformidade

FROM workspace_db_case_espec_dados_riscos.gold_obsolescencia_por_servidor

GROUP BY
    servidor,
    sigla_origem;