CREATE OR REPLACE VIEW workspace_db_case_espec_dados_riscos.vw_qtd_servidores_por_farol AS

WITH base AS (

    SELECT *
    FROM workspace_db_case_espec_dados_riscos.vw_percentual_inconformidade_por_servidor

)

SELECT
    CASE
        WHEN percentual_inconformidade <= 5
            THEN 'Verde'

        WHEN percentual_inconformidade > 5
         AND percentual_inconformidade < 20
            THEN 'Amarelo'

        ELSE 'Vermelho'
    END AS farol,

    COUNT(*) AS qtd_servidores,

    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentual_servidores

FROM base

GROUP BY 1

ORDER BY
    CASE
        WHEN farol = 'Verde' THEN 1
        WHEN farol = 'Amarelo' THEN 2
        ELSE 3
    END;