# pymarrmot
A Python version of the Modular Assessment of Rainfall-Runoff Models Toolbox (MARRMoT)
https://github.com/wknoben/MARRMoT

Following conversion from Matlab, models were tested against results from Matlab using a range of parameterizations (p05, p25, p50,	p75, p95 of range prescribed by W. Knoben) and forcings (daily timestep). Results were identical or near-identical for all models with the following exceptions:

The following models exhibited different behavior across all parameterizations:
m_43_gsmsocont_12p_6s
m_44_echo_16p_6s

The following models exhibited different behavior when parameters were set to 5th percentile of the acceptable range, as prescribed by w. Knoben:
m_07_gr4j_4p_2s
m_08_us1_5p_2s
m_16_newzealand2_8p_2s
m_20_gsfb_8p_3s
m_25_tcm_6p_4s
m_33_sacramento_11p_5s
m_37_hbv_15p_5s
m_43_gsmsocont_12p_6s
m_44_echo_16p_6s


