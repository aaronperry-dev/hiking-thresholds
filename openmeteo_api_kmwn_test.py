import requests
import pandas as pd
import numpy as np
import json
import argparse
import urllib.parse
from datetime import datetime, timedelta, timezone
import sys

def elevation_to_pressure_hpa(elevation_m):
    if elevation_m is None or elevation_m <= 0:
        return 800.0
    p0 = 1013.25
    t0 = 288.15
    lapse_rate = 0.0065
    p_target = p0 * ((1.0 - (lapse_rate * elevation_m) / t0) ** 5.25588)
    return round(float(p_target), 1)

def geocode_location(location_query):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location_query)}&count=1&language=en&format=json"
    resp = requests.get(url)
    
    if resp.status_code == 200:
        data = resp.json()
        if "results" in data and len(data["results"]) > 0:
            res = data["results"][0]
            lat = res.get("latitude")
            lon = res.get("longitude")
            elevation = res.get("elevation")
            admin = res.get("admin1", res.get("country", ""))
            name_full = f"{res.get('name')}, {admin}".strip(", ")
            return lat, lon, elevation, name_full
    return None, None, None, location_query

def calculate_wind_chill(t, v):
    wc = np.where(
        (t <= 50) & (v >= 3),
        35.74 + (0.6215 * t) - (35.75 * (v ** 0.16)) + (0.4275 * t * (v ** 0.16)),
        t
    )
    return wc

def log_interpolate(v_lower, v_upper, p_lower, p_upper, p_target):
    if p_lower == p_upper:
        return v_lower
    ratio = (np.log(p_target) - np.log(p_lower)) / (np.log(p_upper) - np.log(p_lower))
    return v_lower + (v_upper - v_lower) * ratio

def align_to_master(model_times, model_data, master_times):
    aligned = np.full(len(master_times), np.nan)
    if not model_times or not model_data:
        return aligned
    data_dict = dict(zip(model_times, model_data))
    for i, mt in enumerate(master_times):
        if mt in data_dict and data_dict[mt] is not None:
            aligned[i] = data_dict[mt]
    return aligned

def fetch_dynamic_levels(model, lat, lon, start_date, end_date, target_p):
    candidate_levels = [950, 925, 900, 850, 825, 800, 775, 750, 700, 600, 500]
    valid_data = {}
    
    for level in candidate_levels:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&start_date={start_date}&end_date={end_date}"
               f"&hourly=temperature_{level}hPa,wind_speed_{level}hPa"
               f"&models={model}&wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=America/New_York")
        resp = requests.get(url)
        if resp.status_code == 200:
            hourly = resp.json().get("hourly", {})
            t_data = hourly.get(f"temperature_{level}hPa")
            w_data = hourly.get(f"wind_speed_{level}hPa")
            if t_data and any(x is not None for x in t_data):
                valid_data[level] = {
                    "t": t_data,
                    "w": w_data,
                    "times": hourly.get("time")
                }
    
    if len(valid_data) == 0:
        return None, None, None
        
    levels = sorted(valid_data.keys())
    
    if int(target_p) in valid_data:
        p = int(target_p)
        return valid_data[p]["t"], valid_data[p]["w"], valid_data[p]["times"]
        
    if len(valid_data) == 1:
        only_p = levels[0]
        return valid_data[only_p]["t"], valid_data[only_p]["w"], valid_data[only_p]["times"]

    lower_p = max([p for p in levels if p >= target_p], default=None)
    upper_p = min([p for p in levels if p < target_p], default=None)
    
    if lower_p is None: lower_p, upper_p = levels[0], levels[1]
    if upper_p is None: lower_p, upper_p = levels[-2], levels[-1]

    t_lower = np.array(valid_data[lower_p]["t"], dtype=float)
    t_upper = np.array(valid_data[upper_p]["t"], dtype=float)
    w_lower = np.array(valid_data[lower_p]["w"], dtype=float)
    w_upper = np.array(valid_data[upper_p]["w"], dtype=float)
    
    t_interp = log_interpolate(t_lower, t_upper, lower_p, upper_p, target_p)
    w_interp = log_interpolate(w_lower, w_upper, lower_p, upper_p, target_p)
    
    return t_interp.tolist(), w_interp.tolist(), valid_data[lower_p]["times"]

def fetch_model_precipitation(model, lat, lon, elevation_m, start_date, end_date):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&elevation={elevation_m}"
           f"&start_date={start_date}&end_date={end_date}"
           f"&hourly=precipitation,rain,snowfall,weather_code"
           f"&models={model}&precipitation_unit=inch&timezone=America/New_York")
    resp = requests.get(url)
    if resp.status_code == 200:
        hourly = resp.json().get("hourly", {})
        return (
            hourly.get("precipitation"),
            hourly.get("rain"),
            hourly.get("snowfall"),
            hourly.get("weather_code"),
            hourly.get("time")
        )
    
    url_fallback = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&elevation={elevation_m}"
                    f"&start_date={start_date}&end_date={end_date}"
                    f"&hourly=precipitation,weather_code"
                    f"&models={model}&precipitation_unit=inch&timezone=America/New_York")
    resp_fb = requests.get(url_fallback)
    if resp_fb.status_code == 200:
        hourly = resp_fb.json().get("hourly", {})
        return (
            hourly.get("precipitation"),
            None,
            None,
            hourly.get("weather_code"),
            hourly.get("time")
        )
    return None, None, None, None, None

def classify_precipitation_types(precip_arr, rain_arr, snow_arr, wcode_arr, temp_arr):
    n = len(precip_arr)
    rain_out = np.zeros(n)
    snow_out = np.zeros(n)
    mixed_out = np.zeros(n)
    
    for i in range(n):
        p = precip_arr[i]
        if np.isnan(p) or p <= 0.001:
            continue
            
        t = temp_arr[i] if (temp_arr is not None and not np.isnan(temp_arr[i])) else 32.0
        wcode = wcode_arr[i] if (wcode_arr is not None and wcode_arr[i] is not None) else None
        
        if wcode in [56, 57, 66, 67, 79]: 
            mixed_out[i] = p
        elif wcode in [71, 73, 75, 77, 85, 86]:
            snow_out[i] = p
        elif wcode in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            if t <= 32.0:
                mixed_out[i] = p 
            else:
                rain_out[i] = p
        else:
            if t <= 28.0:
                snow_out[i] = p
            elif 28.0 < t <= 32.0:
                r_val = rain_arr[i] if (rain_arr is not None and not np.isnan(rain_arr[i])) else 0.0
                if r_val > 0.005:
                    mixed_out[i] = p
                else:
                    snow_out[i] = p
            elif 32.0 < t <= 34.0:
                s_val = snow_arr[i] if (snow_arr is not None and not np.isnan(snow_arr[i])) else 0.0
                r_val = rain_arr[i] if (rain_arr is not None and not np.isnan(rain_arr[i])) else 0.0
                if s_val > 0.005 and r_val > 0.005:
                    mixed_out[i] = p
                elif s_val > 0.005:
                    snow_out[i] = p
                else:
                    rain_out[i] = p
            else:
                rain_out[i] = p

    return rain_out, snow_out, mixed_out

def fetch_kmwn_mos(cycle, elevation_m, master_times_api):
    now_utc = datetime.now(timezone.utc)
    candidate_dates = [now_utc.strftime("%Y%m%d"), (now_utc - timedelta(days=1)).strftime("%Y%m%d"), (now_utc - timedelta(days=2)).strftime("%Y%m%d")]
    
    resp, date_str = None, None
    for d in candidate_dates:
        url_mex = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs_mos/prod/gfs_mos.{d}/mdl_gfsmex.t{cycle}z"
        resp_mex = requests.get(url_mex, timeout=10)
        if resp_mex.status_code == 200:
            resp = resp_mex
            date_str = d
            break
            
        url_mav = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs_mos/prod/gfs_mos.{d}/mdl_gfsmav.t{cycle}z"
        resp_mav = requests.get(url_mav, timeout=10)
        if resp_mav.status_code == 200:
            resp = resp_mav
            date_str = d
            break
            
    if resp is None:
        return None, None
        
    block = []
    in_block = False
    for line in resp.text.split('\n'):
        if line.strip().startswith("KMWN") and "MOS GUIDANCE" in line:
            in_block = True
        if in_block:
            block.append(line)
            if line.strip() == "" and len(block) > 5:
                break
                
    if not block: return None, None
    
    hr_line = next((l for l in block if l.startswith("HR ") or l.startswith("FHR")), None)
    tmp_line = next((l for l in block if l.startswith("TMP")), None)
    wsp_line = next((l for l in block if l.startswith("WSP")), None)
    
    if not (hr_line and tmp_line and wsp_line):
        return None, None
        
    is_fhr = "FHR" in hr_line
    mos_times, mos_temps, mos_winds = [], [], []
    base_dt = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=int(cycle))
    current_dt = base_dt
    last_hr = -1
    
    hr_cols = hr_line[4:].replace('|', ' ').split()
    tmp_cols = tmp_line[4:].replace('|', ' ').split()
    wsp_cols = wsp_line[4:].replace('|', ' ').split()
    
    for i in range(min(len(hr_cols), len(tmp_cols), len(wsp_cols))):
        if not hr_cols[i].isdigit(): continue
        hr_val = int(hr_cols[i])
        
        if is_fhr: 
            step_dt = base_dt + timedelta(hours=hr_val)
        else:      
            if hr_val < last_hr:
                current_dt += timedelta(days=1)
            step_dt = current_dt.replace(hour=hr_val)
            last_hr = hr_val
            
        mos_times.append(step_dt.strftime('%Y-%m-%dT%H:%M'))
        
        t_str = tmp_cols[i]
        mos_temps.append(float(t_str) if t_str.lstrip('-').isdigit() else np.nan)
        
        w_str = wsp_cols[i]
        mos_winds.append(float(w_str) * 1.15078 if w_str.isdigit() else np.nan)
            
    delta_m = elevation_m - 1917
    lapse_adj_f = delta_m * -0.0117
    adj_temps = [t + lapse_adj_f if not np.isnan(t) else np.nan for t in mos_temps]
    
    df_mos = pd.DataFrame({"time": mos_times, "t": adj_temps, "w": mos_winds})
    df_mos['time'] = pd.to_datetime(df_mos['time'])
    df_mos.set_index('time', inplace=True)
    
    target_idx = pd.to_datetime(master_times_api)
    df_interp = df_mos.reindex(df_mos.index.union(target_idx)).interpolate(method='time').reindex(target_idx)
    
    return df_interp['t'].tolist(), df_interp['w'].tolist()

def build_mwo_text_forecast_html(dates, res_t, res_w, res_g, res_c, res_p, rain, snow, mixed):
    html = "<div class='card'><h3>Text Forecast</h3>"
    periods = {}
    
    for i, d_str in enumerate(dates):
        dt = datetime.strptime(d_str, '%Y-%m-%dT%H:%M')
        if 6 <= dt.hour < 18:
            p_name = dt.strftime('%A')
        else:
            if dt.hour < 6:
                p_name = (dt - timedelta(days=1)).strftime('%A Night')
            else:
                p_name = dt.strftime('%A Night')
                
        if p_name not in periods:
            periods[p_name] = {'idx': []}
        periods[p_name]['idx'].append(i)
        
    for p_name, data in periods.items():
        idx = data['idx']
        if not idx: continue
        
        t_med = res_t.loc[idx, 'MEDIAN'].values
        
        w_med = res_w.loc[idx, 'MEDIAN'].values
        g_med = res_g.loc[idx, 'MEDIAN'].values
        
        c_med = res_c.loc[idx, 'MEDIAN'].values
        
        p_med = res_p.loc[idx, 'MEDIAN'].values
        p_90 = res_p.loc[idx, 'P90'].values
        
        s_sum = sum(snow[i] for i in idx)
        m_sum = sum(mixed[i] for i in idx)
        r_sum = sum(rain[i] for i in idx)
        p90_sum = np.nansum(p_90)
        pmed_sum = np.nansum(p_med)
        
        weather = "In and out of the clouds under partly sunny skies." if 'Night' not in p_name else "In the clear under partly cloudy skies."
        precip_types = []
        if m_sum > 0.02: precip_types.append("a wintry mix")
        if r_sum > 0.02: precip_types.append("rain")
        if s_sum > 0.02: precip_types.append("snow showers")
        
        if precip_types:
            weather = "Mostly in the clouds w/ a chance of " + " and ".join(precip_types) + "."
            if s_sum > 0.02:
                ratio = (p90_sum / pmed_sum) if pmed_sum > 0 else 2.0
                s_max = s_sum * ratio
                if s_max < 0.1: s_max = 0.1
                weather += f" Possible snow accumulations of {s_sum:.1f} to {s_max:.1f} inches."
            
        t_max_med = np.nanmax(t_med)
        t_min_med = np.nanmin(t_med)
        t_start = t_med[0]
        t_end = t_med[-1]
        
        if 'Night' in p_name:
            if t_end > t_start + 5:
                temp_str = f"Low: Starting near {t_min_med:.0f}°F then rising to {t_end:.0f}°F"
            else:
                temp_str = f"Low: Around {t_min_med:.0f}°F"
        else:
            if t_end < t_start - 5:
                temp_str = f"High: Starting near {t_max_med:.0f}°F then falling to {t_end:.0f}°F"
            else:
                temp_str = f"High: Around {t_max_med:.0f}°F"
            
        w_min = 5 * round(np.nanmin(w_med)/5)
        w_max = 5 * round(np.nanmax(w_med)/5)
        g_max = 5 * round(np.nanmax(g_med)/5)
        
        if w_max == w_min: w_max += 5
        
        wind_str = f"Winds {w_min:.0f}-{w_max:.0f} mph w/ gusts up to {g_max:.0f} mph."
        
        def fmt_c(val):
            return f"{abs(val):.0f} below" if val < 0 else f"{val:.0f} above"
            
        c_min = np.nanmin(c_med)
        c_max = np.nanmax(c_med)
        
        chill_str = f"{fmt_c(c_min)} to {fmt_c(c_max)}."
        
        html += f"<h4>{p_name}:</h4><p style='margin: 4px 0 6px; font-size: 0.95em;'>{weather}</p>"
        html += f"<ul style='margin-top: 0; margin-bottom: 12px; padding-left: 20px; font-size: 0.95em;'>"
        
        parts = temp_str.split(':', 1)
        html += f"<li><strong>{parts[0]}:</strong> {parts[1].strip() if len(parts) > 1 else ''}</li>"
        html += f"<li><strong>Wind:</strong> {wind_str}</li>"
        html += f"<li><strong>Wind Chill:</strong> {chill_str}</li></ul>"
        
    html += "</div>"
    return html

def generate_dashboard(master_times_api, hc_display_times, res_t, res_w, res_g, res_c, res_p, df_t, df_w, df_g, df_c, df_p, df_ens_t, df_ens_w, df_ens_g, df_ens_c, df_ens_p, blended_rain, blended_snow, blended_mixed, date_label, location_name, elevation_m, target_p, successful_models, ens_count):
    mwo_forecast_html = build_mwo_text_forecast_html(
        master_times_api, res_t, res_w, res_g, res_c, res_p, blended_rain, blended_snow, blended_mixed
    )

    def to_hc_series(df, name_prefix, dash_style='Solid', median_color='#000000', decimals=1):
        series = []
        for col in df.columns:
            data = [None if pd.isna(x) else round(x, decimals) for x in df[col].tolist()]
            lw, c, zi = (3, median_color, 10) if col == 'MEDIAN' else (1, '#cccccc', 1)
            series.append({"name": f"{name_prefix} {col}", "data": data, "dashStyle": dash_style, "lineWidth": lw, "color": c, "zIndex": zi})
        return series

    model_colors = {"ncep_gfs_seamless": "#E91E63", "dwd_seamless": "#8BC34A", "ecmwf_ifs04": "#9C27B0",
                    "gem_global": "#4CAF50", "icon_global": "#FFC107", "ecmwf_aifs04": "#673AB7",
                    "ukmo_seamless": "#00BCD4", "jma_seamless": "#F44336", "hrrr_conus": "#8BC34A",
                    "rap_conus": "#FF9800", "gfs_mos_kmwn": "#FF5722"}
                    
    model_names = {"ncep_gfs_seamless": "GFS", "dwd_seamless": "DWD", "ecmwf_ifs04": "ECMWF",
                   "gem_global": "GEM", "icon_global": "ICON", "ecmwf_aifs04": "AIFS",
                   "ukmo_seamless": "UKMET", "jma_seamless": "JMA", "hrrr_conus": "HRRR",
                   "rap_conus": "RAP", "gfs_mos_kmwn": "GFS MOS (Extrapolated)"}

    def to_hc_raw_series(df, metric_name, dash_style='Dot', decimals=1):
        series = []
        for col in df.columns:
            data = [None if pd.isna(x) else round(x, decimals) for x in df[col].tolist()]
            series.append({"name": f"Raw {model_names.get(col, col)} ({metric_name})", "data": data, "dashStyle": dash_style, "lineWidth": 2, "color": model_colors.get(col, '#000000'), "visible": False, "zIndex": 5})
        return series

    def to_hc_ens_series(df, metric_name, dash_style='LongDash', decimals=1):
        series = []
        ens_colors = {
            "GEFS Mean": "#E91E63", 
            "EPS Mean": "#9C27B0", 
            "GEPS Mean": "#4CAF50",
            "DWD ENS Mean": "#FFC107",
            "GDMENS Mean": "#00BCD4"
        }
        for col in df.columns:
            data = [None if pd.isna(x) else round(x, decimals) for x in df[col].tolist()]
            series.append({"name": f"{col} ({metric_name})", "data": data, "dashStyle": dash_style, "lineWidth": 2, "color": ens_colors.get(col, '#000000'), "visible": False, "zIndex": 6})
        return series

    combined_temp = to_hc_series(res_t, "Temp", median_color="#FF5722") + to_hc_raw_series(df_t, "Temp") + to_hc_ens_series(df_ens_t, "Temp")
    combined_wind = to_hc_series(res_w, "Wind Speed", "Solid", "#03A9F4") + to_hc_series(res_g, "Wind Gust", "ShortDash", "#0277BD") + to_hc_raw_series(df_w, "Speed", "Solid") + to_hc_raw_series(df_g, "Gust", "ShortDash") + to_hc_ens_series(df_ens_w, "Speed", "LongDash") + to_hc_ens_series(df_ens_g, "Gust", "LongDash")
    combined_chill = to_hc_series(res_c, "Chill", median_color="#673AB7") + to_hc_raw_series(df_c, "Chill") + to_hc_ens_series(df_ens_c, "Chill")

    precip_stacked_types = [
        {"type": "column", "name": "Snow (Liquid Equiv)", "data": [round(x, 2) for x in blended_snow], "stack": "precip", "color": "#64B5F6", "zIndex": 4},
        {"type": "column", "name": "Mixed (Sleet / Freezing Rain)", "data": [round(x, 2) for x in blended_mixed], "stack": "precip", "color": "#BA68C8", "zIndex": 4},
        {"type": "column", "name": "Rain", "data": [round(x, 2) for x in blended_rain], "stack": "precip", "color": "#26A69A", "zIndex": 4}
    ]
    combined_precip = precip_stacked_types + to_hc_series(res_p, "Total Precip", median_color="#004D40", decimals=2) + to_hc_raw_series(df_p, "Precip", decimals=2) + to_hc_ens_series(df_ens_p, "Precip", decimals=2)

    elev_ft = int(elevation_m * 3.28084) if elevation_m else "N/A"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{location_name} Summit Dashboard - {date_label}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highcharts/11.3.0/highcharts.js"></script>
    <style>
        body {{ font-family: Arial; background: #f4f4f4; padding: 20px; }} 
        .card {{ max-width: 1200px; margin: 0 auto 20px; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }} 
        .chart-container {{ height: 500px; }}
        h4 {{ margin-bottom: 2px; color: #333; font-size: 1.1em; }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">{location_name} Summit Forecast ({date_label})</h1>
    <div class="card">
        <h3>Summit & Diagnostic Metadata</h3>
        <p><strong>Summit Elevation:</strong> {elevation_m}m ({elev_ft} ft)</p>
        <p><strong>Derived Target Atmospheric Pressure:</strong> {target_p} hPa (ISA Standard Profile)</p>
        <p><strong>Deterministic Models Blended:</strong> {", ".join([model_names.get(m, m) for m in successful_models])}</p>
        <p><strong>Ensembles Blended:</strong> {ens_count} total members (GEFS, GEPS, EPS, DWD ENS, GDMENS)</p>
        <p style="color: #555; font-size: 0.9em;"><em>Note: All models and ensemble members are log-interpolated to the exact summit elevation pressure profile ({target_p} hPa). Precipitation is partitioned into Rain, Snow, and Mixed based on atmospheric temperature profiles and WMO weather codes. GFS MOS extrapolated based on ISA lapse rate from 1917m.</em></p>
    </div>
    
    {mwo_forecast_html}

    <div class="card chart-container" id="temp-chart"></div>
    <div class="card chart-container" id="wind-chart"></div>
    <div class="card chart-container" id="chill-chart"></div>
    <div class="card chart-container" id="precip-chart"></div>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            const opts = {{ chart: {{ type: 'line' }}, xAxis: {{ categories: {json.dumps(hc_display_times)} }}, tooltip: {{ shared: true, valueDecimals: 1 }}, credits: {{ enabled: false }} }};
            
            Highcharts.chart('temp-chart', Highcharts.merge(opts, {{ 
                title: {{ text: 'Temperature (°F)' }}, 
                yAxis: {{ 
                    title: {{ text: 'Temperature (°F)' }},
                    plotBands: [
                        {{ from: 10, to: 32, color: 'rgba(3, 169, 244, 0.1)', label: {{ text: 'Freezing', align: 'left', style: {{ color: '#666' }} }} }},
                        {{ from: -100, to: 10, color: 'rgba(3, 169, 244, 0.25)', label: {{ text: 'Extreme Cold', align: 'left', style: {{ color: '#333', fontWeight: 'bold' }} }} }}
                    ]
                }},
                series: {json.dumps(combined_temp)} 
            }}));
            
            Highcharts.chart('wind-chart', Highcharts.merge(opts, {{ 
                title: {{ text: 'Wind & Gusts (mph)' }}, 
                yAxis: {{ 
                    title: {{ text: 'Speed (mph)' }},
                    plotBands: [
                        {{ from: 35, to: 55, color: 'rgba(255, 193, 7, 0.2)', label: {{ text: 'Caution: Walking difficult', align: 'left', style: {{ color: '#666' }} }} }},
                        {{ from: 55, to: 300, color: 'rgba(244, 67, 54, 0.2)', label: {{ text: 'Danger: Blown over risk', align: 'left', style: {{ color: '#333', fontWeight: 'bold' }} }} }}
                    ]
                }},
                tooltip: {{
                    shared: true,
                    useHTML: true,
                    formatter: function () {{
                        let s = '<b>' + this.x + '</b><br/>';
                        let speed = '<div style="margin-top:5px;"><strong>Wind Speed</strong><br/>';
                        let gust = '<div style="margin-top:5px; padding-top:5px; border-top:1px solid #ccc;"><strong>Wind Gust</strong><br/>';
                        this.points.forEach(function (point) {{
                            let line = '<span style="color:' + point.series.color + '">●</span> ' + point.series.name + ': <b>' + point.y.toFixed(1) + '</b><br/>';
                            if (point.series.name.includes('Speed')) {{ speed += line; }}
                            else if (point.series.name.includes('Gust')) {{ gust += line; }}
                        }});
                        return s + speed + '</div>' + gust + '</div>';
                    }}
                }},
                series: {json.dumps(combined_wind)} 
            }}));
            
            Highcharts.chart('chill-chart', Highcharts.merge(opts, {{ 
                title: {{ text: 'Wind Chill (°F)' }}, 
                yAxis: {{ 
                    title: {{ text: 'Wind Chill (°F)' }},
                    plotBands: [
                        {{ from: -19, to: 0, color: 'rgba(255, 193, 7, 0.2)', label: {{ text: 'Caution: Frostbite in 30 mins', align: 'left', style: {{ color: '#666' }} }} }},
                        {{ from: -100, to: -19, color: 'rgba(244, 67, 54, 0.2)', label: {{ text: 'Danger: Rapid frostbite (< 30 mins)', align: 'left', style: {{ color: '#333', fontWeight: 'bold' }} }} }}
                    ]
                }},
                series: {json.dumps(combined_chill)} 
            }}));

            Highcharts.chart('precip-chart', Highcharts.merge(opts, {{ 
                title: {{ text: 'Precipitation & Precipitation Types (in.)' }}, 
                yAxis: {{ 
                    title: {{ text: 'Precipitation (in.)' }},
                    min: 0,
                    plotBands: [
                        {{ from: 0.10, to: 0.25, color: 'rgba(255, 193, 7, 0.15)', label: {{ text: 'Caution: Moderate Precipitation (Soaking)', align: 'left', style: {{ color: '#666' }} }} }},
                        {{ from: 0.25, to: 10.0, color: 'rgba(244, 67, 54, 0.15)', label: {{ text: 'Danger: Heavy Precipitation (Severe Exposure / Whiteout)', align: 'left', style: {{ color: '#333', fontWeight: 'bold' }} }} }}
                    ]
                }},
                plotOptions: {{
                    column: {{
                        stacking: 'normal',
                        groupPadding: 0.05,
                        pointPadding: 0.02,
                        borderWidth: 0
                    }},
                    line: {{
                        marker: {{ enabled: false }}
                    }}
                }},
                tooltip: {{
                    shared: true,
                    useHTML: true,
                    valueDecimals: 2,
                    formatter: function () {{
                        let s = '<b>' + this.x + '</b><br/>';
                        let types = '<div style="margin-top:5px;"><strong>Precipitation Types (Blended)</strong><br/>';
                        let total = '<div style="margin-top:5px; padding-top:5px; border-top:1px solid #ccc;"><strong>Total Percentiles & Models</strong><br/>';
                        
                        this.points.forEach(function (point) {{
                            let line = '<span style="color:' + point.series.color + '">●</span> ' + point.series.name + ': <b>' + point.y.toFixed(2) + ' in.</b><br/>';
                            if (point.series.type === 'column') {{
                                types += line;
                            }} else {{
                                total += line;
                            }}
                        }});
                        return s + types + '</div>' + total + '</div>';
                    }}
                }},
                series: {json.dumps(combined_precip)} 
            }}));
        }});
    </script>
</body>
</html>"""
    with open("forecast_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("\nDashboard successfully exported to forecast_dashboard.html\n")

def get_blended_forecast(start_date, end_date, location_query, mos_cycle_input):
    date_label = start_date if start_date == end_date else f"{start_date} to {end_date}"
    print(f"Resolving location: {location_query}")
    lat, lon, elevation_m, location_name = geocode_location(location_query)
    
    if lat is None or lon is None:
        print(f"Error: Could not resolve coordinates for '{location_query}'. Using Mount Washington defaults.")
        lat, lon, elevation_m = 44.2706, -71.3033, 1917
        location_name = "Mount Washington, NH"
        
    target_p = elevation_to_pressure_hpa(elevation_m)
    print(f"Target Summit: {location_name} (Elevation: {elevation_m}m / {int(elevation_m * 3.28084)}ft)")
    print(f"Calculated Summit Pressure Level: {target_p} hPa\n")
    
    master_dates = pd.date_range(start=f"{start_date} 00:00", end=f"{end_date} 23:00", freq='h')
    master_times_api = master_dates.strftime('%Y-%m-%dT%H:%M').tolist()
    
    if start_date == end_date:
        display_times = master_dates.strftime('%I:%M %p').tolist()
        hc_display_times = master_dates.strftime('%a %I:%M %p').tolist()
        time_col_width = 8
        time_hdr = "Time"
    else:
        display_times = master_dates.strftime('%m/%d %I:%M %p').tolist()
        hc_display_times = master_dates.strftime('%a %m/%d %I:%M %p').tolist()
        time_col_width = 14
        time_hdr = "Date/Time"
    
    deterministic_models = ["ncep_gfs_seamless", "dwd_seamless", "ecmwf_ifs04", "gem_global", 
                            "icon_global", "ecmwf_aifs04", "ukmo_seamless", "jma_seamless", 
                            "hrrr_conus", "rap_conus"]
                            
    ensemble_models = {"ncep_gefs_seamless": "GEFS", "ecmwf_ifs025_ensemble": "EPS", 
                       "gem_global_ensemble": "GEPS", "icon_seamless": "DWD ENS", 
                       "google_weathernext2_ensemble": "GDMENS"}
    
    temps, winds, gusts, chills, precips = {}, {}, {}, {}, {}
    model_rain_dict, model_snow_dict, model_mixed_dict = {}, {}, {}
    successful_models = []
    
    print("Fetching Deterministic Models...\n")
    for m in deterministic_models:
        raw_t, raw_w, model_times = fetch_dynamic_levels(m, lat, lon, start_date, end_date, target_p)
        if raw_t is None:
            print(f"Skipping {m}: No pressure data available for this range/level.")
            continue

        t_aligned = align_to_master(model_times, raw_t, master_times_api)
        w_aligned = align_to_master(model_times, raw_w, master_times_api)
        
        p_raw, r_raw, s_raw, wc_raw, p_times = fetch_model_precipitation(m, lat, lon, elevation_m, start_date, end_date)
        if p_raw is not None:
            p_aligned = align_to_master(p_times, p_raw, master_times_api)
            r_aligned = align_to_master(p_times, r_raw, master_times_api) if r_raw else np.full(len(master_times_api), np.nan)
            s_aligned = align_to_master(p_times, s_raw, master_times_api) if s_raw else np.full(len(master_times_api), np.nan)
            wc_aligned = align_to_master(p_times, wc_raw, master_times_api) if wc_raw else np.full(len(master_times_api), np.nan)
            
            r_typed, s_typed, m_typed = classify_precipitation_types(p_aligned, r_aligned, s_aligned, wc_aligned, t_aligned)
            
            precips[m] = p_aligned
            model_rain_dict[m] = r_typed
            model_snow_dict[m] = s_typed
            model_mixed_dict[m] = m_typed
            print(f"Loaded {m} (Interpolated to {target_p}hPa + Precipitation Typed)")
        else:
            precips[m] = np.full(len(master_times_api), np.nan)
            print(f"Loaded {m} (Interpolated to {target_p}hPa)")

        temps[m], winds[m] = t_aligned, w_aligned
        gusts[m], chills[m] = w_aligned * 1.4, calculate_wind_chill(t_aligned, w_aligned)
        successful_models.append(m)

    print("\nFetching GFS MOS Guidance (KMWN)...")
    mos_t, mos_w = fetch_kmwn_mos(mos_cycle_input, elevation_m, master_times_api)
    if mos_t and mos_w and not all(np.isnan(x) for x in mos_t):
        print(f"Successfully loaded GFS MOS {mos_cycle_input}Z (Extrapolated {elevation_m - 1917:+.1f}m from KMWN)")
        mos_key = "gfs_mos_kmwn"
        temps[mos_key] = mos_t
        winds[mos_key] = mos_w
        gusts[mos_key] = [w * 1.4 if not np.isnan(w) else np.nan for w in mos_w]
        chills[mos_key] = calculate_wind_chill(np.array(mos_t), np.array(mos_w)).tolist()
        precips[mos_key] = np.full(len(master_times_api), np.nan) 
        successful_models.append(mos_key)
    else:
        print(f"GFS MOS {mos_cycle_input}Z not available or failed to parse for target dates.")

    print("\nFetching Ensemble Members from Open-Meteo API...")
    ens_keys_total = []
    
    for ens_id, ens_name in ensemble_models.items():
        url_ens = (f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}&elevation={elevation_m}"
                   f"&start_date={start_date}&end_date={end_date}"
                   f"&hourly=temperature_850hPa,temperature_700hPa,wind_speed_850hPa,wind_speed_700hPa,precipitation"
                   f"&models={ens_id}&precipitation_unit=inch&wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=America/New_York")
        
        resp = requests.get(url_ens)
        if resp.status_code == 200:
            ens_hourly = resp.json().get("hourly", {})
            ens_times = ens_hourly.get("time", [])
            count = 0
            
            member_keys = [k for k in ens_hourly.keys() if k.startswith("temperature_850hPa_member")]
            for t850_key in member_keys:
                member_num = t850_key.replace("temperature_850hPa_", "")
                t700_key = f"temperature_700hPa_{member_num}"
                w850_key = f"wind_speed_850hPa_{member_num}"
                w700_key = f"wind_speed_700hPa_{member_num}"
                p_key = f"precipitation_{member_num}"
                
                t_850 = np.array(ens_hourly.get(t850_key, []), dtype=float)
                w_850 = np.array(ens_hourly.get(w850_key, [np.nan]*len(t_850)), dtype=float)
                p_val = np.array(ens_hourly.get(p_key, [np.nan]*len(t_850)), dtype=float)
                
                raw_t700 = ens_hourly.get(t700_key)
                if raw_t700 is None or all(x is None for x in raw_t700):
                    t_interp = t_850
                else:
                    t_700 = np.array(raw_t700, dtype=float)
                    t_interp = log_interpolate(t_850, t_700, 850, 700, target_p)
                    
                raw_w700 = ens_hourly.get(w700_key)
                if raw_w700 is None or all(x is None for x in raw_w700):
                    w_interp = w_850
                else:
                    w_700 = np.array(raw_w700, dtype=float)
                    w_interp = log_interpolate(w_850, w_700, 850, 700, target_p)
                
                t_aligned = align_to_master(ens_times, t_interp.tolist(), master_times_api)
                w_aligned = align_to_master(ens_times, w_interp.tolist(), master_times_api)
                p_aligned = align_to_master(ens_times, p_val.tolist(), master_times_api)
                
                dict_key = f"{ens_id}_{member_num}"
                temps[dict_key], winds[dict_key] = t_aligned, w_aligned
                precips[dict_key] = p_aligned
                gusts[dict_key], chills[dict_key] = w_aligned * 1.4, calculate_wind_chill(t_aligned, w_aligned)
                
                count += 1
                ens_keys_total.append(dict_key)
                
            print(f"  Successfully loaded {ens_name} ({count} members, Interpolated to {target_p}hPa)")
        else:
            print(f"  Failed to load {ens_name} (HTTP {resp.status_code})")

    print(f"Total ensemble members blended: {len(ens_keys_total)}")

    df_t, df_w = pd.DataFrame(temps), pd.DataFrame(winds)
    df_g, df_c = pd.DataFrame(gusts), pd.DataFrame(chills)
    df_p = pd.DataFrame(precips)
    
    if "Washington" in location_name:
        print("\nApplying topographical wind multiplier (1.45x) for Mount Washington...")
        df_w = df_w * 1.45
        df_g = df_g * 1.45
        for col in df_c.columns:
            if col in df_t.columns and col in df_w.columns:
                df_c[col] = calculate_wind_chill(df_t[col].values, df_w[col].values)
    
    ens_families = {
        "ncep_gefs_seamless": "GEFS Mean", 
        "ecmwf_ifs025_ensemble": "EPS Mean", 
        "gem_global_ensemble": "GEPS Mean",
        "icon_seamless": "DWD ENS Mean",
        "google_weathernext2_ensemble": "GDMENS Mean"
    }
    ens_means_t, ens_means_w, ens_means_g, ens_means_c, ens_means_p = {}, {}, {}, {}, {}
    
    for fam_key, fam_name in ens_families.items():
        fam_cols = [c for c in df_t.columns if fam_key in str(c) and 'member' in str(c)]
        if fam_cols:
            with np.errstate(all='ignore'):
                ens_means_t[fam_name] = df_t[fam_cols].mean(axis=1, skipna=True)
                ens_means_w[fam_name] = df_w[fam_cols].mean(axis=1, skipna=True)
                ens_means_g[fam_name] = df_g[fam_cols].mean(axis=1, skipna=True)
                ens_means_c[fam_name] = df_c[fam_cols].mean(axis=1, skipna=True)
                if not df_p.empty and any(c in df_p.columns for c in fam_cols):
                    p_cols = [c for c in fam_cols if c in df_p.columns]
                    ens_means_p[fam_name] = df_p[p_cols].mean(axis=1, skipna=True)
                
    df_ens_t, df_ens_w = pd.DataFrame(ens_means_t), pd.DataFrame(ens_means_w)
    df_ens_g, df_ens_c = pd.DataFrame(ens_means_g), pd.DataFrame(ens_means_c)
    df_ens_p = pd.DataFrame(ens_means_p)

    def calc_percentiles(df):
        with np.errstate(all='ignore'):
            return pd.DataFrame(np.nanpercentile(df.values, [0, 10, 25, 50, 50, 75, 90, 100], axis=1, method='weibull').T, columns=['MIN', 'P10', 'P25', 'P50', 'MEDIAN', 'P75', 'P90', 'MAX'])

    res_t, res_w = calc_percentiles(df_t), calc_percentiles(df_w)
    res_g, res_c = calc_percentiles(df_g), calc_percentiles(df_c)
    res_p = calc_percentiles(df_p)
    
    num_timesteps = len(master_times_api)
    blended_rain = np.zeros(num_timesteps)
    blended_snow = np.zeros(num_timesteps)
    blended_mixed = np.zeros(num_timesteps)
    
    if model_rain_dict:
        df_m_rain = pd.DataFrame(model_rain_dict)
        df_m_snow = pd.DataFrame(model_snow_dict)
        df_m_mixed = pd.DataFrame(model_mixed_dict)
        
        for i in range(num_timesteps):
            med_p = res_p.loc[i, 'MEDIAN']
            if pd.isna(med_p) or med_p <= 0.001:
                continue
            
            r_sum = df_m_rain.iloc[i].sum(skipna=True)
            s_sum = df_m_snow.iloc[i].sum(skipna=True)
            m_sum = df_m_mixed.iloc[i].sum(skipna=True)
            total_sum = r_sum + s_sum + m_sum
            
            if total_sum > 0:
                blended_rain[i] = (r_sum / total_sum) * med_p
                blended_snow[i] = (s_sum / total_sum) * med_p
                blended_mixed[i] = (m_sum / total_sum) * med_p
            else:
                blended_rain[i] = med_p

    def fmt(val, is_precip=False):
        if pd.isna(val): return "  NaN"
        return f"{val:5.2f}" if is_precip else f"{val:5.1f}"
        
    def fmt_med(val, is_precip=False):
        if pd.isna(val): return "   NaN"
        return f"{val:6.2f}" if is_precip else f"{val:6.1f}"

    print(f"\nBlended Forecast Percentiles for {location_name} @ {target_p}hPa ({date_label})")
    print("=" * (78 + time_col_width))
    print(f"{time_hdr:>{time_col_width}} {'Metric':>9} {'MIN':>5} {'P10':>5} {'P25':>5} {'P50':>5} | {'MEDIAN':^6} | {'P75':>5} {'P90':>5} {'MAX':>5}")
    
    metrics_data = [
        ("Temp(°F)", res_t, False),
        ("Wind(mph)", res_w, False),
        ("Gust(mph)", res_g, False),
        ("Chill(°F)", res_c, False),
        ("Precip(in)", res_p, True)
    ]
    
    for i in range(len(display_times)):
        for m_idx, (metric_name, df_metric, is_p) in enumerate(metrics_data):
            print(f"{display_times[i] if m_idx == 0 else '':>{time_col_width}} {metric_name:>9} {fmt(df_metric.iloc[i]['MIN'], is_p)} {fmt(df_metric.iloc[i]['P10'], is_p)} {fmt(df_metric.iloc[i]['P25'], is_p)} {fmt(df_metric.iloc[i]['P50'], is_p)} | {fmt_med(df_metric.iloc[i]['MEDIAN'], is_p)} | {fmt(df_metric.iloc[i]['P75'], is_p)} {fmt(df_metric.iloc[i]['P90'], is_p)} {fmt(df_metric.iloc[i]['MAX'], is_p)}")
        print()

    generate_dashboard(
        master_times_api, hc_display_times, res_t, res_w, res_g, res_c, res_p,
        df_t[successful_models], df_w[successful_models], df_g[successful_models], df_c[successful_models], df_p[successful_models],
        df_ens_t, df_ens_w, df_ens_g, df_ens_c, df_ens_p,
        blended_rain, blended_snow, blended_mixed,
        date_label, location_name, elevation_m, target_p, successful_models, len(ens_keys_total)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blended summit weather forecast supporting single dates or multi-day ranges.")
    parser.add_argument("--date", type=str, required=True, help="Forecast date (YYYYMMDD) or range (YYYYMMDD-YYYYMMDD)")
    parser.add_argument("--location", type=str, required=True, help="Location name (e.g., 'Mount Washington, NH')")
    parser.add_argument("--mos_cycle", type=str, choices=["00", "12"], default="00", help="GFS MOS Run Cycle to fetch (00 or 12)")
    args = parser.parse_args()
    
    date_input = args.date.strip()
    try:
        if "-" in date_input and len(date_input.split("-")) == 2:
            s_str, e_str = date_input.split("-")
            start_date = datetime.strptime(s_str.strip(), "%Y%m%d").strftime("%Y-%m-%d")
            end_date = datetime.strptime(e_str.strip(), "%Y%m%d").strftime("%Y-%m-%d")
        else:
            start_date = datetime.strptime(date_input, "%Y%m%d").strftime("%Y-%m-%d")
            end_date = start_date
    except ValueError:
        print("Error: Date must be formatted as YYYYMMDD (e.g. 20261101) or YYYYMMDD-YYYYMMDD (e.g. 20261001-20261003)")
        sys.exit(1)
        
    get_blended_forecast(start_date, end_date, args.location, args.mos_cycle)