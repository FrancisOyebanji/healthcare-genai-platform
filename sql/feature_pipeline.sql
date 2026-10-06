-- Healthcare feature pipeline (Snowflake / Databricks SQL dialect).
-- Builds member-month features from claims, pharmacy, and utilization-management
-- sources for the predictive / prior-authorization models. Illustrative schema;
-- no real data. CPT/HCPCS/ICD-10/DRG/Revenue codes are used as join keys.

-- 1) Normalize medical claims to one row per claim line ------------------------
CREATE OR REPLACE TABLE gold.claim_lines AS
SELECT
    c.claim_id,
    c.member_id,
    c.service_date,
    DATE_TRUNC('month', c.service_date)            AS service_month,
    c.place_of_service,
    c.revenue_code,                                 -- UB-04 revenue code
    c.cpt_hcpcs_code,                               -- CPT-4 / HCPCS
    c.icd10_primary,                                -- ICD-10-CM
    c.drg_code,                                     -- MS-DRG (inpatient)
    c.allowed_amount,
    c.paid_amount
FROM raw.medical_claims c
WHERE c.claim_status = 'PAID'
  AND c.service_date >= DATEADD('month', -24, CURRENT_DATE);

-- 2) Member-month utilization + cost features ---------------------------------
CREATE OR REPLACE TABLE gold.member_month_features AS
WITH util AS (
    SELECT
        member_id,
        service_month,
        COUNT(DISTINCT claim_id)                                   AS claims_cnt,
        COUNT_IF(place_of_service = 'ER')                          AS er_visits,
        COUNT_IF(drg_code IS NOT NULL)                             AS inpatient_lines,
        SUM(paid_amount)                                           AS paid_amt,
        COUNT(DISTINCT icd10_primary)                             AS distinct_dx
    FROM gold.claim_lines
    GROUP BY member_id, service_month
),
chronic AS (   -- chronic-condition flags from ICD-10 prefixes
    SELECT
        member_id,
        service_month,
        MAX(CASE WHEN icd10_primary LIKE 'E11%' THEN 1 ELSE 0 END) AS flag_diabetes,
        MAX(CASE WHEN icd10_primary = 'I10'     THEN 1 ELSE 0 END) AS flag_htn,
        MAX(CASE WHEN icd10_primary LIKE 'N18%' THEN 1 ELSE 0 END) AS flag_ckd,
        MAX(CASE WHEN icd10_primary LIKE 'J44%' THEN 1 ELSE 0 END) AS flag_copd
    FROM gold.claim_lines
    GROUP BY member_id, service_month
)
SELECT
    u.member_id,
    u.service_month,
    u.claims_cnt,
    u.er_visits,
    u.inpatient_lines,
    u.paid_amt,
    u.distinct_dx,
    c.flag_diabetes, c.flag_htn, c.flag_ckd, c.flag_copd,
    -- rolling 3-month cost (window feature)
    SUM(u.paid_amt) OVER (
        PARTITION BY u.member_id ORDER BY u.service_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )                                                             AS paid_amt_3mo
FROM util u
LEFT JOIN chronic c
  ON u.member_id = c.member_id AND u.service_month = c.service_month;

-- 3) Prior-authorization request feed for the agentic reviewer ----------------
--    (requested procedure/drug + the clinical facts the policy criteria need)
CREATE OR REPLACE VIEW gold.pa_request_feed AS
SELECT
    pa.request_id,
    pa.member_id,
    pa.requested_cpt_hcpcs,
    pa.request_text,
    f.flag_diabetes, f.flag_htn, f.flag_ckd, f.flag_copd,
    f.paid_amt_3mo
FROM raw.pa_requests pa
LEFT JOIN gold.member_month_features f
  ON pa.member_id = f.member_id
 AND DATE_TRUNC('month', pa.request_date) = f.service_month
WHERE pa.request_status = 'PENDING_REVIEW';
