"""Fungal-risk 38-feature schema shared by training and NEST telemetry inference."""

# Strictly environmental + 7-day trend predictors (no calendar/site/target leakage)
FEATURE_COLS: list[str] = [
    # Temperature (8)
    "temp_mean_7d", "temp_min_7d", "temp_max_7d", "temp_std_7d",
    "temp_diurnal_range_mean_7d", "temp_mean_24h", "dp_mean_7d", "dp_depression_mean_7d",
    # Humidity (7)
    "rh_mean_7d", "rh_min_7d", "rh_max_7d", "rh_std_7d",
    "rh_mean_24h", "rh_ge_80_hours_7d", "rh_ge_90_hours_7d",
    # Leaf Wetness (7)
    "lw_mean_7d", "lw_max_7d", "lw_mean_24h", "lw_wet_hours_7d",
    "lw_heavy_wet_hours_7d", "lw_wet_hours_24h", "favorable_infection_hours_7d",
    # Rainfall, Soil & Solar (7)
    "rain_total_7d", "rain_max_daily_7d", "rain_days_count_7d", "rain_total_24h",
    "soil_moisture_mean_7d", "soil_moisture_mean_24h", "solar_radiation_mean_7d",
    # 7-Day Trends (9)
    "temp_trend_slope_7d", "rh_trend_slope_7d", "lw_hours_trend_slope_7d",
    "rain_trend_slope_7d", "soil_moisture_trend_slope_7d",
    "temp_delta_3d_vs_4d", "rh_delta_3d_vs_4d", "lw_hours_delta_3d_vs_4d",
    "rain_delta_3d_vs_4d",
]
