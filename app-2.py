import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------------------------
# STREAMLIT SETUP & CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(page_title="DIEM Quality Control Dashboard", layout="wide")
st.title("📊 DIEM-Monitoring - Global QC Dashboard Template")
st.markdown("Replicates all mandatory and optional DIEM quality checks with exact SPSS precision.")

DK_REF_VALS = [888, 999, '888', '999', 888.0, 999.0]

# ---------------------------------------------------------------------
# FLAG EXPLANATIONS
# ---------------------------------------------------------------------
FLAG_EXPLANATIONS = {
    "survey_num_sum": "Indicates the total number of surveys conducted per enumerator.",
    "submission_flag": "Indicates the mean of the total number of flags received per enumerator.",
    "flag_long_survey": "Indicates the average proportion of surveys lasting for more than 60 minutes.",
    "flag_z_survey_time": "Indicates the average proportion of surveys lasting for too long or too short (± 2 std dev).",
    "flag_z_survey_num": "Indicates the average proportion of surveys where enumerator conducted too many or too little interviews per day (± 2 std dev).",
    "flag_dkref": "Indicates the average proportion of surveys where 'don't know' or 'refused' are frequently selected (≥ 2 std dev).",
    "flag_other": "Indicates the average proportion of surveys where 'other' is frequently selected (≥ 2 std dev).",
    "flag_trigger": "Indicates the average proportion of surveys where no, don't know or N/A are frequently selected for key variables (≥ 2 std dev).",
    "flag_outliers": "Indicates the average proportion of surveys where numeric variables contain outliers (≥ 2 std dev).",
    "flag_crop_hh": "Household reports receiving income from crop production but indicates not being engaged in crop production.",
    "flag_harv": "Household reports a decrease in harvest but does not report any production difficulty.",
    "flag_crop_price": "Household reports a decrease in crop prices but does not report any input issues (in terms of prices).",
    "flag_crop_price_increase": "Household reports an increase in crop prices and reports an input issue (in terms of low selling prices).",
    "flag_livestock_hh": "Household reports receiving income from livestock production but also reports not being engaged in it.",
    "flag_ls_num_diff": "Outlier in the difference of livestock number since last year (in tropical livestock units).",
    "flag_liv_change": "Household reports a decrease in number of livestock but does not report any cause for decrease.",
    "flag_liv_price": "Household reports a decrease in livestock prices but does not report any input issues (in terms of prices).",
    "flag_liv_price_increase": "Household reports an increase in livestock prices and reports an input issue (in terms of low selling prices).",
    "flag_hdds_fewfood": "Household reports a high HDDS score (>9), but reports eating only a few kinds of foods due to lack of money.",
    "flag_low_hdds_score": "Household dietary diversity score is unreasonably low (<= 2). May indicate low probing.",
    "flag_fies_severe": "Household relies on severe coping strategies before exhausting mild coping strategies.",
    "flag_insuff_probe": "More than 50% of multiple-choice (MC) questions are answered with a single response (low probing).",
    "flag_income_amount": "Household reported ‘0’ as income amount despite having an income source.",
    "flag_fish_hh": "Household reports receiving income from fish production and indicates not being engaged in it.",
    "flag_fish_price_increase": "Household reports an increase in fish prices and reports an input issue (low selling prices).",
    "flag_sold_animals": "Inconsistencies between selling animals as a stress coping strategy vs the reasons and number of animals sold.",
    "flag_borrowed": "Inconsistencies where a household doesn't need to borrow money but reports using exclusively savings/debt.",
    "flag_agric_input": "Households did not indicate difficulties accessing agricultural inputs but had to decrease expenditure on them as a coping strategy.",
    "flag_migration": "Households report permanent residence but also declare having migrated with their entire family recently.",
    "flag_non_agric_hh": "Households reported using an agricultural coping strategy but did not report being an agricultural household.",
    "flag_fies_skipped_rcsi_reduced": "Inconsistencies between not missing a meal due to lack of money vs reducing the number of daily meals.",
    "flag_hdds_fcs": "Inconsistencies between not consuming a food group in the last 7 days vs having consumed it yesterday."
}

# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------
def safe_numeric(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(np.nan, index=df.index)

def calculate_z_scores(series):
    s_num = pd.to_numeric(series.replace(DK_REF_VALS, np.nan), errors='coerce')
    std = s_num.std()
    if pd.notna(std) and std != 0:
        return (s_num - s_num.mean()) / std
    return pd.Series(0, index=series.index)

def parse_duration_to_minutes(series):
    s_num = pd.to_numeric(series, errors='coerce')
    if not s_num.isna().all():
        return s_num
    s_td = pd.to_timedelta(series, errors='coerce')
    return s_td.dt.total_seconds() / 60.0

# ---------------------------------------------------------------------
# MAIN FLAGGING ENGINE (STRICT SPSS MATCHING)
# ---------------------------------------------------------------------
def generate_flags(df, settings):
    df = df.copy()
    flags = pd.DataFrame(index=df.index)
    flags['survey_num'] = 1
    flags['submission_flag'] = 0

    # 1. ENUMERATOR HANDLING
    if 'enumerator' not in df.columns:
        if 'operator_id' in df.columns: df['enumerator'] = df['operator_id']
        elif 'deviceid' in df.columns: df['enumerator'] = df['deviceid']
        else: df['enumerator'] = 'Unknown'
    df['enumerator'] = df['enumerator'].fillna('Unknown')
    flags['enumerator'] = df['enumerator']

    # 2. TIME / DURATION FLAGS
    df['survey_time_mins'] = np.nan
    if settings.get('duration_calc_type') == 'start_end':
        if settings['start_col'] in df.columns and settings['end_col'] in df.columns:
            start_s = pd.to_datetime(df[settings['start_col']], errors='coerce', utc=True)
            end_s = pd.to_datetime(df[settings['end_col']], errors='coerce', utc=True)
            df['survey_time_mins'] = (end_s - start_s).dt.total_seconds() / 60.0
    elif settings.get('duration_col') in df.columns:
        df['survey_time_mins'] = parse_duration_to_minutes(df[settings['duration_col']])
    
    if df['survey_time_mins'].notna().any():
        flags['flag_long_survey'] = np.where(df['survey_time_mins'] > 60, 1, 0)
        flags['flag_z_survey_time'] = np.where((calculate_z_scores(df['survey_time_mins']).abs() > 2), 1, 0)
    else:
        flags['flag_long_survey'] = 0; flags['flag_z_survey_time'] = 0

    # 3. SURVEYS PER DAY 
    if settings['date_col'] in df.columns:
        df['today_date'] = pd.to_datetime(df[settings['date_col']], errors='coerce', utc=True).dt.date
        surveys_per_day = df.groupby(['today_date', 'enumerator']).size().reset_index(name='daily_count')
        surveys_per_day['Z_daily_count'] = calculate_z_scores(surveys_per_day['daily_count'])
        surveys_per_day['flag_z_survey_num'] = np.where(surveys_per_day['Z_daily_count'].abs() > 2, 1, 0)
        df = df.merge(surveys_per_day[['today_date', 'enumerator', 'flag_z_survey_num']], on=['today_date', 'enumerator'], how='left')
        flags['flag_z_survey_num'] = df['flag_z_survey_num'].fillna(0)
    else:
        flags['flag_z_survey_num'] = 0

    # 4. EXACT DON'T KNOW / REFUSED & OTHER LOGIC
    dkref_888_999 = [
        'resp_gender', 'hh_agricactivity', 'hh_gender', 'hh_education', 'resp_id', 'hh_size', 'hh_maritalstat', 
        'hh_residencetype', 'hh_age', 'hh_wealth_water', 'hh_wealth_toilet', 'hh_wealth_light', 'income_main', 
        'income_sec', 'income_third', 'income_main_amount', 'income_sec_amount', 'income_third_amount', 
        'income_main_gender', 'income_main_control', 'income_main_comp', 'income_sec_comp', 'income_third_comp', 
        'crp_landsize_num', 'crp_main', 'crp_irrig_source', 'crp_area_change', 'crp_harv_change', 'crp_proddif', 
        'crp_salesmain', 'crp_salesdif', 'crp_salesprice', 'crp_harv_vol', 'crp_harv_unit_kg', 'crp_harv_lastyr', 
        'crp_storage', 'ls_main', 'ls_num_lastyr', 'ls_num_now', 'ls_proddif', 'ls_salesmain', 'ls_salesdif', 
        'ls_salesprice', 'fish_change', 'fish_proddif', 'fish_salesmain', 'fish_salesdif', 'fish_salesprice', 
        'fies_worried', 'fies_healthy', 'fies_fewfoods', 'fies_skipped', 'fies_ateless', 'fies_ranout', 
        'fies_ranout_hhs', 'fies_hungry', 'fies_hungry_hhs', 'fies_whlday', 'fies_whlday_hhs', 'need', 
        'assistance_quality', 'cs_stress_hh_assets', 'cs_stress_spent_savings', 'cs_stress_sold_more_animals', 
        'cs_stress_borrowed_or_helped', 'cs_stress_credit', 'cs_stress_borrowed_money', 'cs_crisis_sold_prod_assets', 
        'cs_crisis_no_school', 'cs_crisis_reduced_health_exp', 'cs_crisis_consume_seed_stock', 
        'cs_crisis_decrease_input_exp', 'cs_emergency_begged', 'cs_emergency_illegal', 'cs_emergency_hh_migration'
    ]
    dkref_1 = [
        'shock_dk', 'shock_ref', 'crp_proddif_dk', 'crp_proddif_ref', 'crp_saledif_dk', 'crp_saledif_ref', 
        'crp_seed_dk', 'crp_seed_ref', 'ls_num_inc_dec_dk', 'ls_num_inc_dec_ref', 'ls_proddif_dk', 'ls_proddif_ref', 
        'ls_salesdif_dk', 'ls_salesdif_ref', 'ls_feed_dk', 'ls_feed_ref', 'fish_main_dk', 'fish_main_ref', 
        'fish_proddif_dk', 'fish_proddif_ref', 'fish_saledif_dk', 'fish_saledif_ref', 'fish_inputdif_dk', 
        'fish_inputdif_ref', 'need_dk', 'need_ref', 'need_received_dk', 'need_received_ref', 'assistance_dk', 
        'assistance_ref'
    ]
    
    dk888_exist = [c for c in dkref_888_999 if c in df.columns]
    dk1_exist = [c for c in dkref_1 if c in df.columns]
    
    df['dkref_count'] = df[dk888_exist].isin(DK_REF_VALS).sum(axis=1) + df[dk1_exist].isin([1, '1', 1.0]).sum(axis=1)
    flags['flag_dkref'] = np.where(calculate_z_scores(df['dkref_count']) >= 2, 1, 0)

    # EXACT OTHER LOGIC
    other_1 = [c for c in df.columns if c.endswith('_other') or c.endswith('_Otherspecify_other')]
    df['other_count'] = df[other_1].isin([1, '1', 1.0]).sum(axis=1)
    if 'crp_irrig_source' in df.columns: df['other_count'] += np.where(safe_numeric(df, 'crp_irrig_source') == 6, 1, 0)
    if 'crp_harv_unit' in df.columns: df['other_count'] += np.where(safe_numeric(df, 'crp_harv_unit') == 9, 1, 0)
    if 'ls_main' in df.columns: df['other_count'] += np.where(safe_numeric(df, 'ls_main') == 9, 1, 0)
    if 'fish_salesmain' in df.columns: df['other_count'] += np.where(safe_numeric(df, 'fish_salesmain') == 5, 1, 0)
    flags['flag_other'] = np.where(calculate_z_scores(df['other_count']) >= 2, 1, 0)

    # 5. EXACT SKIP TRIGGERS
    skip_1 = [c for c in ['hh_agricactivity'] if c in df.columns]
    skip_2 = [c for c in ['fies_ranout', 'fies_hungry', 'fies_whlday', 'crp_proddif', 'crp_salesdif', 'ls_proddif', 'ls_salesdif', 'fish_proddif', 'fish_salesdif'] if c in df.columns]
    skip_3 = [c for c in ['income_main', 'income_sec', 'income_third', 'crp_main', 'crp_salesmain', 'ls_main', 'ls_salesmain'] if c in df.columns]
    
    df['trigger_count'] = df[skip_1].isin([4, 888, 999, '4', '888', '999']).sum(axis=1) + \
                          df[skip_2].isin([0, 888, 999, '0', '888', '999']).sum(axis=1) + \
                          df[skip_3].isin([888, 999, '888', '999']).sum(axis=1)
    flags['flag_trigger'] = np.where(calculate_z_scores(df['trigger_count']) >= 2, 1, 0)

    # 6. NUMERICAL OUTLIERS
    outlier_cols = ['crp_landsize_num', 'ls_num_lastyr', 'ls_num_now', 'hh_size', 'income_main_amount', 'income_sec_amount', 'income_third_amount', 'crp_harv_vol', 'crp_harv_unit_kg', 'crp_harv_lastyr', 'fcs_staple_days', 'fcs_pulses_days', 'fcs_vegetables_days', 'fcs_fruit_days', 'fcs_meat_fish_days', 'fcs_dairy_days', 'fcs_sugar_days', 'fcs_oil_days', 'fcs_condiments_days', 'rcsi_less_preferred_foods', 'rcsi_borrowed_food', 'rcsi_reduce_number_meals', 'rcsi_limit_portions', 'rcsi_restrict_adult_consumpt']
    avail_outliers = [c for c in outlier_cols if c in df.columns]
    df['num_outlier_count'] = 0
    for col in avail_outliers:
        z = calculate_z_scores(df[col])
        df['num_outlier_count'] += np.where((z <= -2) | (z > 2), 1, 0)
    flags['flag_outliers'] = np.where(calculate_z_scores(df['num_outlier_count']) >= 2, 1, 0)

    # 7. LOGICAL CHECKS - INCOME
    inc_main = safe_numeric(df, 'income_main')
    inc_sec = safe_numeric(df, 'income_sec')
    inc_third = safe_numeric(df, 'income_third')
    
    flags['flag_income_amount'] = np.where(
        ((safe_numeric(df, 'income_main_amount') == 0) & (inc_main <= 18)) |
        ((safe_numeric(df, 'income_sec_amount') == 0) & (inc_sec <= 18)) |
        ((safe_numeric(df, 'income_third_amount') == 0) & (inc_third <= 18)), 1, 0)
    
    hh_agric = safe_numeric(df, 'hh_agricactivity')
    crop_inc = (inc_main <= 3) | (inc_sec <= 3) | (inc_third <= 3)
    flags['flag_crop_hh'] = np.where(crop_inc & (~hh_agric.isin([1, 3])), 1, 0)
    
    liv_inc = (inc_main == 4) | (inc_sec == 4) | (inc_third == 4)
    flags['flag_livestock_hh'] = np.where(liv_inc & (~hh_agric.isin([2, 3])), 1, 0)
    
    fish_inc = (inc_main == 6) | (inc_sec == 6) | (inc_third == 6)
    flags['flag_fish_hh'] = np.where(fish_inc & (safe_numeric(df, 'hh_fish') != 1), 1, 0)

    # 8. LOGICAL CHECKS - CROP
    harv_change = safe_numeric(df, 'crp_harv_change')
    crp_salesprice = safe_numeric(df, 'crp_salesprice')
    factor_low_crop = np.where((safe_numeric(df, 'crp_saledif_marketing_cost') == 1) | (safe_numeric(df, 'crp_saledif_low_demand') == 1) | (safe_numeric(df, 'crp_saledif_low_price') == 1), 1, 0)
    
    flags['flag_harv'] = np.where((harv_change.isin([4,5])) & (safe_numeric(df, 'crp_proddif') == 0), 1, 0)
    flags['flag_crop_price'] = np.where((crp_salesprice.isin([4,5])) & (factor_low_crop == 0), 1, 0)
    flags['flag_crop_price_increase'] = np.where((crp_salesprice.isin([1,2])) & (safe_numeric(df, 'crp_saledif_low_price') == 1), 1, 0)

    # 9. LOGICAL CHECKS - LIVESTOCK
    ls_main = safe_numeric(df, 'ls_main')
    ls_diff = safe_numeric(df, 'ls_num_now').replace(DK_REF_VALS, np.nan) - safe_numeric(df, 'ls_num_lastyr').replace(DK_REF_VALS, np.nan)
    
    # Tropical units
    ls_diff = np.where(ls_main == 1, ls_diff * 0.5, ls_diff)
    ls_diff = np.where(ls_main.isin([2,3]), ls_diff * 0.1, ls_diff)
    ls_diff = np.where(ls_main.isin([6,7]), ls_diff * 0.01, ls_diff)
    ls_diff = np.where(ls_main == 4, ls_diff * 0.2, ls_diff)
    ls_diff = np.where(ls_main.isin([5,8]), ls_diff * 0.01, ls_diff)
    
    flags['flag_ls_num_diff'] = np.where(calculate_z_scores(pd.Series(ls_diff)).abs() > 2, 1, 0)

    ls_price = safe_numeric(df, 'ls_salesprice')
    factor_low_liv = np.where((safe_numeric(df, 'ls_salesdif_marketing_cost') == 1) | (safe_numeric(df, 'ls_salesdif_low_demand') == 1) | (safe_numeric(df, 'ls_salesdif_low_price') == 1), 1, 0)
    flags['flag_liv_price'] = np.where((ls_price.isin([4,5])) & (factor_low_liv == 0), 1, 0)
    flags['flag_liv_price_increase'] = np.where((ls_price.isin([1,2])) & (safe_numeric(df, 'ls_salesdif_low_price') == 1), 1, 0)
    
    reason_dec = (safe_numeric(df, 'ls_num_dec_poor_health').fillna(0) + safe_numeric(df, 'ls_num_dec_death').fillna(0) + 
                  safe_numeric(df, 'ls_num_dec_sales_good_price').fillna(0) + safe_numeric(df, 'ls_num_dec_sales_distress').fillna(0) + 
                  safe_numeric(df, 'ls_num_dec_escape_stolen').fillna(0) + safe_numeric(df, 'ls_num_dec_consumed').fillna(0) + 
                  safe_numeric(df, 'ls_num_inc_dec_other').fillna(0))
    flags['flag_liv_change'] = np.where((pd.Series(ls_diff) < 0) & (reason_dec == 0), 1, 0)

    # 10. LOGICAL CHECKS - FISHERIES
    flags['flag_fish_price_increase'] = np.where((safe_numeric(df, 'fish_salesprice').isin([1,2])) & (safe_numeric(df, 'fish_saledif_low_prices') == 1), 1, 0)

    # 11. LOGICAL CHECKS - FOOD SEC, HDDS, COPING STRATEGIES
    hdds_cols = [c for c in df.columns if c.startswith('hdds_')]
    if hdds_cols:
        hdds_score = df[hdds_cols].apply(pd.to_numeric, errors='coerce').replace(DK_REF_VALS, np.nan).sum(axis=1, min_count=1)
        flags['flag_low_hdds_score'] = np.where(hdds_score <= 2, 1, 0)
        flags['flag_hdds_fewfood'] = np.where((safe_numeric(df, 'fies_fewfoods') == 1) & (hdds_score > 9), 1, 0)
        
        flag_hdds_fcs = np.where(
            ((safe_numeric(df, 'hdds_cereals') == 1) & (safe_numeric(df, 'fcs_staple_days') == 0)) |
            ((safe_numeric(df, 'hdds_rootstubers') == 1) & (safe_numeric(df, 'fcs_staple_days') == 0)) |
            ((safe_numeric(df, 'hdds_vegetables') == 1) & (safe_numeric(df, 'fcs_vegetables_days') == 0)) |
            ((safe_numeric(df, 'hdds_fruits') == 1) & (safe_numeric(df, 'fcs_fruit_days') == 0)) |
            (((safe_numeric(df, 'hdds_meat') == 1) | (safe_numeric(df, 'hdds_eggs') == 1) | (safe_numeric(df, 'hdds_fish') == 1)) & (safe_numeric(df, 'fcs_meat_fish_days') == 0)) |
            ((safe_numeric(df, 'hdds_legumes') == 1) & (safe_numeric(df, 'fcs_pulses_days') == 0)) |
            ((safe_numeric(df, 'hdds_milkdairy') == 1) & (safe_numeric(df, 'fcs_dairy_days') == 0)) |
            ((safe_numeric(df, 'hdds_oils') == 1) & (safe_numeric(df, 'fcs_oil_days') == 0)) |
            ((safe_numeric(df, 'hdds_sugar') == 1) & (safe_numeric(df, 'fcs_sugar_days') == 0)) |
            ((safe_numeric(df, 'hdds_condiments') == 1) & (safe_numeric(df, 'fcs_condiments_days') == 0)), 
            1, 0)
        flags['flag_hdds_fcs'] = flag_hdds_fcs
    else:
        flags['flag_low_hdds_score'] = 0; flags['flag_hdds_fewfood'] = 0; flags['flag_hdds_fcs'] = 0

    fies_mild = safe_numeric(df, 'fies_worried').replace(DK_REF_VALS, np.nan).fillna(0) + safe_numeric(df, 'fies_healthy').replace(DK_REF_VALS, np.nan).fillna(0) + safe_numeric(df, 'fies_fewfoods').replace(DK_REF_VALS, np.nan).fillna(0)
    fies_severe = safe_numeric(df, 'fies_ranout').replace(DK_REF_VALS, np.nan).fillna(0) + safe_numeric(df, 'fies_hungry').replace(DK_REF_VALS, np.nan).fillna(0) + safe_numeric(df, 'fies_whlday').replace(DK_REF_VALS, np.nan).fillna(0)
    flags['flag_fies_severe'] = np.where(fies_severe > fies_mild, 1, 0)
    flags['flag_fies_skipped_rcsi_reduced'] = np.where((safe_numeric(df, 'fies_skipped') == 0) & (safe_numeric(df, 'rcsi_reduce_number_meals') > 0), 1, 0)

    cs_sold_anim = safe_numeric(df, 'cs_stress_sold_more_animals')
    ls_distress = safe_numeric(df, 'ls_num_dec_sales_distress')
    flags['flag_sold_animals'] = np.where(
        (cs_sold_anim.isin([1, 3]) & (ls_distress == 0)) | 
        (cs_sold_anim.isin([1, 3]) & (safe_numeric(df, 'ls_num_dec_sales_good_price') == 1)) | 
        (cs_sold_anim.isin([1, 3]) & ((safe_numeric(df, 'ls_num_inc_less_sales') == 1) | (safe_numeric(df, 'ls_num_no_change') == 1))) |
        (cs_sold_anim.isin([1, 3]) & (safe_numeric(df, 'ls_num_inc_more_acquired') == 1) & (ls_distress == 0)) |
        (cs_sold_anim.isin([888, 999]) | ls_main.isin([888, 999])), 1, 0)

    flags['flag_borrowed'] = np.where((safe_numeric(df, 'cs_stress_borrowed_money').isin([2, 888, 999])) & (inc_main == 19), 1, 0)
    
    access_input = np.where(
        (safe_numeric(df, 'crp_proddif_access_fertilize') == 1) | (safe_numeric(df, 'crp_proddif_seed_quantity') == 1) |
        (safe_numeric(df, 'crp_proddif_access_pesticide') == 1) | (safe_numeric(df, 'crp_proddif_access_labour') == 1) |
        (safe_numeric(df, 'ls_proddif_vet_serv') == 1) | (safe_numeric(df, 'ls_proddif_vet_input') == 1) |
        (safe_numeric(df, 'ls_proddif_feed_purchase') == 1), 1, 0)
    flags['flag_agric_input'] = np.where((safe_numeric(df, 'cs_crisis_decrease_input_exp') == 1) & (access_input == 0), 1, 0)
    
    flags['flag_non_agric_hh'] = np.where(
        (cs_sold_anim.isin([1, 2, 3]) | safe_numeric(df, 'cs_crisis_consume_seed_stock').isin([1, 2, 3]) | safe_numeric(df, 'cs_crisis_decrease_input_exp').isin([1, 2, 3])) & (hh_agric == 4), 1, 0)
    
    mig = safe_numeric(df, 'cs_emergency_hh_migration')
    res = safe_numeric(df, 'hh_residencetype')
    flags['flag_migration'] = np.where(((mig.isin([1, 3])) & (res == 1)) | (mig.isin([888, 999]) | res.isin([888, 999])), 1, 0)

    # 13. INSUFFICIENT PROBING
    mc_prefixes = ['hh_asset_prod_', 'hh_asset_transp_', 'shock_', 'crp_seed_', 'crp_proddif_', 'crp_saledif_', 'ls_num_', 'ls_feed_', 'ls_proddif_', 'ls_salesdif_', 'fish_main_', 'fish_proddif_', 'fish_inputdif_', 'fish_saledif_', 'hdds_', 'need_', 'need_received_', 'assistance_', 'future_int_day_', 'future_int_time_']
    mc_single_count = pd.Series(0, index=df.index)
    mc_valid_count = pd.Series(0, index=df.index)
    
    for prefix in mc_prefixes:
        cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith(('dk', 'ref', 'other', 'none', 'noshock'))]
        if cols:
            selected = df[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            mc_single_count += np.where(selected == 1, 1, 0)
            mc_valid_count += np.where(selected > 0, 1, 0) 
            
    flags['flag_insuff_probe'] = np.where((mc_valid_count > 0) & ((mc_single_count / mc_valid_count) > 0.5), 1, 0)

    # Calculate final submission flag based on user selection
    active_flags = [f for f in flags.columns if f.startswith('flag_') and settings.get(f, True)]
    flags['submission_flag'] = flags[active_flags].sum(axis=1)

    return flags

# ---------------------------------------------------------------------
# UI & DASHBOARD RENDERING
# ---------------------------------------------------------------------
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload GeoPoll/Kobo Data", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        data_sheet_match = [s for s in sheet_names if s.strip().lower() == 'data']
        default_sheet = data_sheet_match[0] if data_sheet_match else sheet_names[0]
        selected_sheet = st.sidebar.selectbox("Data Sheet:", sheet_names, index=sheet_names.index(default_sheet))
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    
    df.columns = df.columns.str.lower().str.strip()

    st.sidebar.header("2. Configuration")
    
    date_col = st.sidebar.selectbox("Date Column (for surveys per day):", df.columns, index=df.columns.tolist().index('survey_created_date') if 'survey_created_date' in df.columns else (df.columns.tolist().index('today') if 'today' in df.columns else 0))
    
    duration_calc_type = st.sidebar.radio("How is survey duration stored?", ["Single Duration Column (GeoPoll)", "Start & End Time Columns (Kobo)"])
    settings = {'date_col': date_col}
    
    if duration_calc_type == "Single Duration Column (GeoPoll)":
        settings['duration_calc_type'] = 'single'
        settings['duration_col'] = st.sidebar.selectbox("Duration Column:", df.columns, index=df.columns.tolist().index('total_case_duration') if 'total_case_duration' in df.columns else 0)
    else:
        settings['duration_calc_type'] = 'start_end'
        settings['start_col'] = st.sidebar.selectbox("Start Time Column:", df.columns, index=df.columns.tolist().index('start') if 'start' in df.columns else 0)
        settings['end_col'] = st.sidebar.selectbox("End Time Column:", df.columns, index=df.columns.tolist().index('end') if 'end' in df.columns else 0)

    with st.sidebar.expander("⚙️ Select Flags to Include"):
        st.write("Toggle which checks contribute to the analysis.")
        flag_options = [f for f in FLAG_EXPLANATIONS.keys() if f.startswith('flag_')]
        for f in flag_options:
            settings[f] = st.checkbox(f, value=True)

    with st.spinner('Calculating ALL Quality Control Flags...'):
        flags_df = generate_flags(df, settings)
        
        # Remove inactive flags from flags_df entirely
        inactive_flags = [f for f in flag_options if not settings.get(f, True)]
        flags_df = flags_df.drop(columns=inactive_flags, errors='ignore')
        
        df_processed = pd.concat([df.drop(columns=['enumerator'], errors='ignore'), flags_df], axis=1)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Quick Report & Insights", "📑 Enumerator Flag Table", "📈 MC Visualizations", "ℹ️ Flag Definitions", "🗃️ Raw Data"])

    # Extract dynamic list of active flag columns
    active_flag_cols = [c for c in flags_df.columns if c.startswith('flag_')]

    # --- TAB 1: QUICK REPORT ---
    with tab1:
        st.subheader("Data Quality Insights")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Surveys", len(flags_df))
        col2.metric("Total Enumerators", flags_df['enumerator'].nunique())
        col3.metric("Avg Flags per Survey", round(flags_df['submission_flag'].mean(), 2))

        st.markdown("---")
        
        # Extract Top Issue
        if active_flag_cols:
            dataset_flag_means = flags_df[active_flag_cols].mean().sort_values(ascending=False)
            top_flag = dataset_flag_means.index[0]
            top_flag_val = dataset_flag_means.iloc[0] * 100

            st.markdown(f"### 🚨 Top Dataset-Wide Issue: `{top_flag}`")
            st.warning(f"**Explanation:** {FLAG_EXPLANATIONS.get(top_flag, 'No explanation found.')}")
            st.write(f"This issue occurred in **{top_flag_val:.1f}%** of all collected surveys. **Action Required:** Discuss this module with the implementation partner.")
        else:
            st.info("No flags selected for analysis.")
            
        st.markdown("---")
        
        # 1. High Risk Enumerators 
        st.markdown("### 🕵️ High-Risk Enumerators (≥ 2.0 Flags/Survey)")
        enum_means = flags_df.groupby('enumerator')['submission_flag'].mean().sort_values(ascending=False)
        problem_enums = enum_means[enum_means >= 2.0]
        
        if not problem_enums.empty:
            st.error(f"Found {len(problem_enums)} enumerators averaging 2.0 or more flags per survey. Consider checking their audio files.")
            st.dataframe(problem_enums.reset_index().rename(columns={"submission_flag": "Average Flags"}).style.background_gradient(cmap='Reds'))
        else:
            st.success("No enumerators are averaging 2.0 or more flags per survey.")

        st.markdown("---")

        # 2. Top 10 Problematic Enumerators
        st.markdown("### 📊 Top 10 Enumerators Issue Breakdown")
        
        if active_flag_cols:
            enum_all_means = flags_df.groupby('enumerator')[active_flag_cols + ['submission_flag']].mean()
            top_10_enums = enum_all_means.sort_values(by='submission_flag', ascending=False).head(10)
            
            if top_10_enums['submission_flag'].max() > 0:
                st.write("Below are the 10 enumerators with the highest average flags per survey, along with a breakdown of their most frequent specific errors:")
                
                for enum, row in top_10_enums.iterrows():
                    avg_flags = row['submission_flag']
                    
                    ind_flags = row.drop('submission_flag').sort_values(ascending=False)
                    ind_flags = ind_flags[ind_flags > 0]
                    
                    top_issues_text = ""
                    if not ind_flags.empty:
                        top_3 = ind_flags.head(3)
                        issues_list = []
                        for flag_name, val in top_3.items():
                            explanation = FLAG_EXPLANATIONS.get(flag_name, "No explanation available.")
                            issues_list.append(f"- **{flag_name}** ({val*100:.1f}% of their surveys): *{explanation}*")
                        top_issues_text = "\n".join(issues_list)
                    else:
                        top_issues_text = "- No specific individual flags recorded."
                    
                    with st.expander(f"Enumerator: **{enum}** | Avg Flags per Survey: **{avg_flags:.2f}**"):
                        st.markdown("**Most Common Issues:**")
                        st.markdown(top_issues_text)
            else:
                st.success("🎉 Excellent! No flags recorded for any enumerators.")

    # --- TAB 2: ENUMERATOR FLAG TABLE ---
    with tab2:
        st.subheader("Enumerator Quality Control Recap")
        
        if active_flag_cols:
            agg_dict = {'survey_num': 'sum', 'submission_flag': 'mean'}
            agg_dict.update({flag: 'mean' for flag in active_flag_cols})
            
            enumerator_summary = flags_df.groupby('enumerator').agg(agg_dict).reset_index()
            enumerator_summary.rename(columns={'survey_num': 'Total Surveys'}, inplace=True)
            
            format_cols = ['submission_flag'] + active_flag_cols
            
            for col in format_cols:
                enumerator_summary[col] = (enumerator_summary[col]).round(3)
                
            st.dataframe(enumerator_summary.style.background_gradient(cmap='Reds', subset=format_cols), height=600, use_container_width=True)
        else:
            st.info("No flags selected to display.")

    # --- TAB 3: VISUALIZATIONS ---
    with tab3:
        st.subheader("Multiple Choice Frequencies")
        mc_prefixes = {'Reported Shocks': 'shock_', 'Crop Difficulties': 'crp_proddif_', 'Dietary Diversity': 'hdds_', 'Needs': 'need_'}
        for chart_title, prefix in mc_prefixes.items():
            valid_cols = [c for c in df_processed.columns if c.startswith(prefix) and not c.endswith(('dk','ref','other'))]
            if valid_cols:
                sums = df_processed[valid_cols].apply(pd.to_numeric, errors='coerce').sum().reset_index()
                sums.columns = ['Option', 'Count']
                sums = sums.sort_values(by='Count', ascending=False)
                fig = px.bar(sums, x='Option', y='Count', title=f'{chart_title} Frequencies', color='Count', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 4: EXPLANATIONS ---
    with tab4:
        st.subheader("Flag Definitions")
        table_md = "| Flag Type | Explanation |\n| :--- | :--- |\n"
        for k, v in FLAG_EXPLANATIONS.items(): table_md += f"| **{k}** | {v} |\n"
        st.markdown(table_md)

    # --- TAB 5: RAW DATA & EXPORT ---
    with tab5:
        st.subheader("Processed Dataset")
        st.dataframe(df_processed.head(100), use_container_width=True)
        csv = df_processed.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Full Processed Data (CSV)", data=csv, file_name='QC_Flagged_Data.csv', mime='text/csv')

else:
    st.info("👈 Please upload your raw data in the sidebar to begin.")
