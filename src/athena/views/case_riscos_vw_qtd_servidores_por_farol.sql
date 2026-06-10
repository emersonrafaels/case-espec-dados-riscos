CREATE OR REPLACE VIEW workspace_db.case_riscos_vw_qtd_servidores_por_farol AS
WITH score_servidor AS (
    SELECT
        servidor,
        sigla_origem,
        MAX(
            CASE
                WHEN indice_obsolescencia > 0
                  OR lower(rating) IN ('alto', 'médio', 'medio', 'baixo')
                THEN 1
                ELSE 0
            END
        ) AS flag_inconforme
    FROM workspace_db.case_riscos_gold_obsolescencia_por_servidor
    GROUP BY
        servidor,
        sigla_origem
),

farol_servidor AS (
    SELECT
        servidor,
        sigla_origem,
        CASE
            WHEN flag_inconforme = 0 THEN 'Verde'
            ELSE 'Vermelho'
        END AS farol
    FROM score_servidor
)

SELECT
    farol,
    COUNT(DISTINCT servidor) AS qtd_servidores
FROM farol_servidor
GROUP BY farol;