import requests
import pandas as pd
import numpy as np
import json
import argparse
import urllib.parse
from datetime import datetime
import sys

def elevation_to_pressure_hpa(elevation_m):
    """
    Converts height above sea level in meters to atmospheric pressure in hPa
    using the International Standard Atmosphere (ISA) barometric equation.
    """
    if elevation_m is None or elevation_m <= 0:
        return 800.0  # Fallback
    p0 = 1013.25  # Standard sea level pressure in hPa
    t0 = 288.15   # Standard sea level temperature in Kelvin (15°C)
    lapse_rate = 0.0065  # Standard lapse rate (K/m)
    p_target = p0 * ((1.0 - (lapse_rate * elevation_m) / t0) ** 5.25588)
    return round(float(p_target), 1)

def geocode_location(location_query):
    """Uses Open-Meteo's Geocoding API to resolve coordinates and true summit elevation."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location_query)}&count=1&language=en&format=json"
    resp = requests.get(url)
    
    if resp.status_code == 200:
        data = resp.json()
        if "results" in data and len(data["results"]) > 0:
            res = data["results"][0]
            lat = res.get("latitude")
            lon = res.get("longitude")
            elevation = res.get("elevation")  # Summit elevation in meters
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
    """Logarithmic interpolation/extrapolation weighted by the summit's target pressure."""
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

def fetch_dynamic_levels(model, lat, lon, target_date, target_p):
    """Fetches pressure levels around target_p and performs log interpolation."""
    candidate_levels = [950, 925, 900, 850, 825, 800, 775, 750, 700, 600, 500]
    valid_data = {}
    
    for level in candidate_levels:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&start_date={target_date}&end_date={target_date}"
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
    
    # If the target pressure level happens to be directly available
    if int(target_p) in valid_data:
        p = int(target_p)
        return valid_data[p]["t"], valid_data[p]["w"], valid_data[p]["times"]
        
    if len(valid_data) == 1:
        only_p = levels[0]
        return valid_data[only_p]["t"], valid_data[only_p]["w"], valid_data[only_p]["times"]

    # Select closest bracketing levels
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

def generate_dashboard(display_times, res_t, res_w, res_g, res_c, df_t, df_w, df_g, df_c, df_ens_t, df_ens_w, df_ens_g, df_ens_c, target_date, location_name, elevation_m, target_p, successful_models, ens_count):
    def to_hc_series(df, name_prefix, dash_style='Solid', median_color='#000000'):
        series = []
        for col in df.columns:
            if col == 'P50': continue
            data = [None if pd.isna(x) else round(x, 1) for x in df[col].tolist()]
            lw, c, zi = (3, median_color, 10) if col == 'MEDIAN' else (1, '#cccccc', 1)
            series.append({"name": f"{name_prefix} {col}", "data": data, "dashStyle": dash_style, "lineWidth": lw, "color": c, "zIndex": zi})
        return series

    model_colors = {"ncep_gfs_seamless": "#E91E63", "ecmwf_ifs04": "#9C27B0", "gem_global": "#4CAF50", "icon_global": "#FFC107", "ecmwf_aifs04": "#673AB7", "ukmo_seamless": "#00BCD4", "jma_seamless": "#F44336", "hrrr_conus": "#8BC34A", "rap_conus": "#FF9800"}
    model_names = {"ncep_gfs_seamless": "GFS", "ecmwf_ifs04": "ECMWF", "gem_global": "GEM", "icon_global": "ICON", "ecmwf_aifs04": "AIFS", "ukmo_seamless": "UKMET", "jma_seamless": "JMA", "hrrr_conus": "HRRR", "rap_conus": "RAP"}

    def to_hc_raw_series(df, metric_name, dash_style='Dot'):
        series = []
        for col in df.columns:
            data = [None if pd.isna(x) else round(x, 1) for x in df[col].tolist()]
            series.append({"name": f"Raw {model_names.get(col, col)} ({metric_name})", "data": data, "dashStyle": dash_style, "lineWidth": 2, "color": model_colors.get(col, '#000000'), "visible": False, "zIndex": 5})
        return series

    def to_hc_ens_series(df, metric_name, dash_style='LongDash'):
        series = []
        ens_colors = {"GEFS Mean": "#E91E63", "EPS Mean": "#9C27B0", "GEPS Mean": "#4CAF50"}
        for col in df.columns:
            data = [None if pd.isna(x) else round(x, 1) for x in df[col].tolist()]
            series.append({"name": f"{col} ({metric_name})", "data": data, "dashStyle": dash_style, "lineWidth": 2, "color": ens_colors.get(col, '#000000'), "visible": False, "zIndex": 6})
        return series

    combined_temp = to_hc_series(res_t, "Temp", median_color="#FF5722") + to_hc_raw_series(df_t, "Temp") + to_hc_ens_series(df_ens_t, "Temp")
    combined_wind = to_hc_series(res_w, "Wind Speed", "Solid", "#03A9F4") + to_hc_series(res_g, "Wind Gust", "ShortDash", "#0277BD") + to_hc_raw_series(df_w, "Speed", "Solid") + to_hc_raw_series(df_g, "Gust", "ShortDash") + to_hc_ens_series(df_ens_w, "Speed", "LongDash") + to_hc_ens_series(df_ens_g, "Gust", "LongDash")
    combined_chill = to_hc_series(res_c, "Chill", median_color="#673AB7") + to_hc_raw_series(df_c, "Chill") + to_hc_ens_series(df_ens_c, "Chill")

    elev_ft = int(elevation_m * 3.28084) if elevation_m else "N/A"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{location_name} Summit Dashboard - {target_date}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highcharts/11.3.0/highcharts.js"></script>
    <style>body {{ font-family: Arial; background: #f4f4f4; padding: 20px; }} .card {{ max-width: 1200px; margin: 0 auto 20px; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }} .chart-container {{ height: 500px; }}</style>
</head>
<body>
    <h1 style="text-align: center;">{location_name} Summit Forecast ({target_date})</h1>
    <div class="card">
        <h3>Summit & Diagnostic Metadata</h3>
        <p><strong>Summit Elevation:</strong> {elevation_m}m ({elev_ft} ft)</p>
        <p><strong>Derived Target Atmospheric Pressure:</strong> {target_p} hPa (ISA Standard Profile)</p>
        <p><strong>Deterministic Models Blended:</strong> {", ".join([model_names.get(m, m) for m in successful_models])}</p>
        <p><strong>Ensembles Blended:</strong> {ens_count} total members (GEFS, GEPS, EPS)</p>
        <p style="color: #555; font-size: 0.9em;"><em>Note: All models and ensemble members are log-interpolated to the exact summit elevation pressure profile ({target_p} hPa).</em></p>
    </div>
    <div class="card chart-container" id="temp-chart"></div>
    <div class="card chart-container" id="wind-chart"></div>
    <div class="card chart-container" id="chill-chart"></div>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            const opts = {{ chart: {{ type: 'line' }}, xAxis: {{ categories: {json.dumps(display_times)} }}, tooltip: {{ shared: true, valueDecimals: 1 }}, credits: {{ enabled: false }} }};
            
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
        }});
    </script>
</body>
</html>"""
    with open("forecast_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("\nDashboard successfully exported to forecast_dashboard.html\n")

def get_blended_forecast(target_date, location_query):
    print(f"Resolving location: {location_query}")
    lat, lon, elevation_m, location_name = geocode_location(location_query)
    
    if lat is None or lon is None:
        print(f"Error: Could not resolve coordinates for '{location_query}'. Using Mount Washington defaults.")
        lat, lon, elevation_m = 44.2706, -71.3033, 1917
        location_name = "Mount Washington, NH"
        
    target_p = elevation_to_pressure_hpa(elevation_m)
    print(f"Target Summit: {location_name} (Elevation: {elevation_m}m / {int(elevation_m * 3.28084)}ft)")
    print(f"Calculated Summit Pressure Level: {target_p} hPa\n")
    
    master_dates = pd.date_range(start=target_date, periods=24, freq='h')
    master_times_api = master_dates.strftime('%Y-%m-%dT%H:%M').tolist()
    display_times = master_dates.strftime('%I:%M %p').tolist()
    
    deterministic_models = ["ncep_gfs_seamless", "ecmwf_ifs04", "gem_global", "icon_global", "ecmwf_aifs04", "ukmo_seamless", "jma_seamless", "hrrr_conus", "rap_conus"]
    ensemble_models = {"gfs_seamless": "GEFS", "ecmwf_ifs025_ensemble": "EPS", "gem_global": "GEPS"}
    
    temps, winds, gusts, chills = {}, {}, {}, {}
    successful_models = []
    
    print("Fetching Deterministic Models...\n")
    for m in deterministic_models:
        raw_t, raw_w, model_times = fetch_dynamic_levels(m, lat, lon, target_date, target_p)
        if raw_t is not None:
            print(f"Loaded {m} (Interpolated to {target_p}hPa)")
        else:
            print(f"Skipping {m}: No data available for this range/level.")
            continue

        t_aligned = align_to_master(model_times, raw_t, master_times_api)
        w_aligned = align_to_master(model_times, raw_w, master_times_api)
        
        temps[m], winds[m] = t_aligned, w_aligned
        gusts[m], chills[m] = w_aligned * 1.4, calculate_wind_chill(t_aligned, w_aligned)
        successful_models.append(m)

    print("\nFetching Ensemble Members from Open-Meteo API...")
    ens_keys_total = []
    
    for ens_id, ens_name in ensemble_models.items():
        url_ens = (f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={lat}&longitude={lon}"
                   f"&start_date={target_date}&end_date={target_date}"
                   f"&hourly=temperature_850hPa,temperature_700hPa,wind_speed_850hPa,wind_speed_700hPa"
                   f"&models={ens_id}&wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=America/New_York")
        
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
                
                t_850 = np.array(ens_hourly.get(t850_key, []), dtype=float)
                w_850 = np.array(ens_hourly.get(w850_key, [np.nan]*len(t_850)), dtype=float)
                
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
                
                dict_key = f"{ens_id}_{member_num}"
                temps[dict_key], winds[dict_key] = t_aligned, w_aligned
                gusts[dict_key], chills[dict_key] = w_aligned * 1.4, calculate_wind_chill(t_aligned, w_aligned)
                
                count += 1
                ens_keys_total.append(dict_key)
                
            print(f"  Successfully loaded {ens_name} ({count} members, Interpolated to {target_p}hPa)")
        else:
            print(f"  Failed to load {ens_name} (HTTP {resp.status_code})")

    print(f"Total ensemble members blended: {len(ens_keys_total)}")

    df_t, df_w = pd.DataFrame(temps), pd.DataFrame(winds)
    df_g, df_c = pd.DataFrame(gusts), pd.DataFrame(chills)
    
    ens_families = {"gfs_seamless": "GEFS Mean", "ecmwf_ifs025_ensemble": "EPS Mean", "gem_global": "GEPS Mean"}
    ens_means_t, ens_means_w, ens_means_g, ens_means_c = {}, {}, {}, {}
    
    for fam_key, fam_name in ens_families.items():
        fam_cols = [c for c in df_t.columns if fam_key in str(c) and 'member' in str(c)]
        if fam_cols:
            with np.errstate(all='ignore'):
                ens_means_t[fam_name] = df_t[fam_cols].mean(axis=1, skipna=True)
                ens_means_w[fam_name] = df_w[fam_cols].mean(axis=1, skipna=True)
                ens_means_g[fam_name] = df_g[fam_cols].mean(axis=1, skipna=True)
                ens_means_c[fam_name] = df_c[fam_cols].mean(axis=1, skipna=True)
                
    df_ens_t, df_ens_w = pd.DataFrame(ens_means_t), pd.DataFrame(ens_means_w)
    df_ens_g, df_ens_c = pd.DataFrame(ens_means_g), pd.DataFrame(ens_means_c)

    def calc_percentiles(df):
        with np.errstate(all='ignore'):
            return pd.DataFrame(np.nanpercentile(df.values, [0, 10, 25, 50, 50, 75, 90, 100], axis=1).T, columns=['MIN', 'P10', 'P25', 'P50', 'MEDIAN', 'P75', 'P90', 'MAX'])

    res_t, res_w = calc_percentiles(df_t), calc_percentiles(df_w)
    res_g, res_c = calc_percentiles(df_g), calc_percentiles(df_c)
    
    def fmt(val): return f"{val:5.1f}" if pd.notna(val) else "  NaN"
    def fmt_med(val): return f"{val:6.1f}" if pd.notna(val) else "   NaN"

    print(f"\nBlended Forecast Percentiles for {location_name} @ {target_p}hPa ({target_date})")
    print("=" * 86)
    print(f"{'Time':>8} {'Metric':>9} {'MIN':>5} {'P10':>5} {'P25':>5} {'P50':>5} | {'MEDIAN':^6} | {'P75':>5} {'P90':>5} {'MAX':>5}")
    
    for i in range(len(display_times)):
        for m_idx, (metric_name, df_metric) in enumerate([("Temp(°F)", res_t), ("Wind(mph)", res_w), ("Gust(mph)", res_g), ("Chill(°F)", res_c)]):
            print(f"{display_times[i] if m_idx == 0 else '':>8} {metric_name:>9} {fmt(df_metric.iloc[i]['MIN'])} {fmt(df_metric.iloc[i]['P10'])} {fmt(df_metric.iloc[i]['P25'])} {fmt(df_metric.iloc[i]['P50'])} | {fmt_med(df_metric.iloc[i]['MEDIAN'])} | {fmt(df_metric.iloc[i]['P75'])} {fmt(df_metric.iloc[i]['P90'])} {fmt(df_metric.iloc[i]['MAX'])}")
        print()

    generate_dashboard(display_times, res_t, res_w, res_g, res_c, df_t[successful_models], df_w[successful_models], df_g[successful_models], df_c[successful_models], df_ens_t, df_ens_w, df_ens_g, df_ens_c, target_date, location_name, elevation_m, target_p, successful_models, len(ens_keys_total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blended weather forecast for a mountain summit.")
    parser.add_argument("--date", type=str, required=True, help="Target date in YYYYMMDD format")
    parser.add_argument("--location", type=str, required=True, help="Location name (e.g., 'Mount Washington, NH')")
    args = parser.parse_args()
    
    try:
        parsed_date = datetime.strptime(args.date, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        print("Error: Date must be in YYYYMMDD format (e.g., 20261101)")
        sys.exit(1)
        
    get_blended_forecast(parsed_date, args.location)