import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pvlib import pvsystem, location, modelchain
from pvlib.iotools import get_pvgis_tmy
from pvlib.solarposition import get_solarposition
from tqdm.contrib.concurrent import process_map
import pvlib.temperature
import functools
import matplotlib.dates as mdates
import multiprocessing
from config import *

def build_horizon_profile(data):
    """
    Creates a pandas Series representing the horizon profile.
    Index is azimuth angles, values are corresponding elevation angles.
    """
    azimuths, elevations = zip(*data)
    return pd.Series(elevations, index=azimuths)

def apply_horizon_mask_to_weather(weather_df, solpos, horizon_profile):
    """
    Applies a horizon mask to DNI (Direct Normal Irradiance) and GHI (Global Horizontal Irradiance)
    in a weather DataFrame based on solar position and horizon profile.
    If the sun is below the horizon obstruction, DNI is set to 0 and GHI is set to DHI (diffuse only).
    """
    # Create a copy to avoid modifying the original DataFrame in place
    masked_weather = weather_df.copy()

    # Interpolate horizon elevation for each solar azimuth angle
    interp_horizon = np.interp(
        solpos['azimuth'].values,
        horizon_profile.index.values,
        horizon_profile.values,
        left=horizon_profile.iloc[0], # Handle values outside range by extending
        right=horizon_profile.iloc[-1]
    )

    # Determine if the sun's elevation is above the horizon obstruction
    above_horizon = solpos['elevation'].values > interp_horizon

    # Apply masking:
    # If below horizon, direct normal irradiance (DNI) becomes 0
    masked_weather['dni'] = np.where(above_horizon, masked_weather['dni'], 0)
    # If below horizon, global horizontal irradiance (GHI) becomes diffuse horizontal irradiance (DHI)
    masked_weather['ghi'] = np.where(above_horizon, masked_weather['ghi'], masked_weather['dhi'])
    # DHI is diffuse and generally not masked by horizon obstructions in this simple model

    return masked_weather

def fill_params(val, param1, param2):
    if val == 'PARAM1':
        return param1
    elif val == 'PARAM2':
        return param2
    else:
        return val

def worker(task, module_params, temp_model_params, inverter_params, loc, masked_weather):
    (param1, param2) = task
    # 2. Define Separate Array Objects for the current orientation.
    # Assuming two arrays as per user's original setup.
    panel1_tilt = fill_params(PANEL_ORIENTATIONS[0]['tilt'], *task)
    panel2_tilt = fill_params(PANEL_ORIENTATIONS[1]['tilt'], *task)
    panel1_azimuth = fill_params(PANEL_ORIENTATIONS[0]['azimuth'], *task)
    panel2_azimuth = fill_params(PANEL_ORIENTATIONS[1]['azimuth'], *task)
    array1 = pvsystem.Array(
        mount=pvsystem.FixedMount(surface_tilt=panel1_tilt, surface_azimuth=panel1_azimuth),
        module_parameters=module_params,
        temperature_model_parameters=temp_model_params,
        modules_per_string=1,
        strings=1,
        name='MPPT1 Array'
    )
    array2 = pvsystem.Array(
        mount=pvsystem.FixedMount(surface_tilt=panel2_tilt, surface_azimuth=panel2_azimuth),
        module_parameters=module_params,
        temperature_model_parameters=temp_model_params,
        modules_per_string=1,
        strings=1,
        name='MPPT2 Array'
    )

    arrays = [array1, array2]

    # 3. Create a PVSystem Object for the current set of arrays and inverter.
    system = pvsystem.PVSystem(
        arrays=arrays,
        inverter_parameters=inverter_params,
        surface_type='urban'
    )

    # 6. Run the PVSystem Model using ModelChain.
    # ModelChain handles the detailed calculations for the entire system.
    mc = modelchain.ModelChain(system, loc,
                                clearsky_model='ineichen', # Used if TMY data has missing values or for comparison
                                temperature_model='sapm',
                                dc_model='cec', # Use CEC DC model for CEC modules
                                ac_model='sandia', # Use Sandia AC model for Sandia/CEC inverters
                                aoi_model='physical') # Angle of Incidence (AOI) model

    # Execute the model simulation with the masked hourly weather data.
    mc.run_model(weather=masked_weather)
    df = mc.results.ac
    df = df[df.index < '2026-01-01'] # Remove 2026-01-01 00:00:00+01:00
    return (param1, param2, df.resample('D').sum() / 1000 * SIM_GRANULARITY_MIN / 60)


def simulate_optimal_orientation(loc, module_params, inverter_params, temp_model_params,
                                 weather, solpos_hourly, horizon_profiles):
    print("Starting annual simulation sweep for optimal orientation...")
    masked_weather = list(map(lambda x: apply_horizon_mask_to_weather(weather, solpos_hourly, x), horizon_profiles))
    tasks = [(param1, param2) for param1 in SIM_PARAM1 for param2 in SIM_PARAM2]
    worker_partial = functools.partial(worker, module_params=module_params,
                                       temp_model_params=temp_model_params, inverter_params=inverter_params,
                                       loc=loc, masked_weather=masked_weather)
    results = process_map(worker_partial, tasks, max_workers=SIM_WORKERS or multiprocessing.cpu_count(), chunksize=1)

    print("Annual simulation sweep complete.")

    annual_matrix, monthly_map = build_energy_matrices(results, SIM_PARAM1, SIM_PARAM2)

    best_param1_annual, best_param2_annual = find_optimal_orientation(annual_matrix)
    annual_matrix.to_csv("csv/annual_energy_matrix.csv")

    monthly_results = compute_monthly_optimal_orientations(monthly_map)
    pd.DataFrame(monthly_results).to_csv("csv/monthly_optimal_orientations.csv", index=False)

    seasonal_results = compute_seasonal_optimal_orientations(results, solpos_hourly)
    # Save just the best orientations (1D data)
    seasonal_summary = {
        'season': ['summer', 'winter'],
        'param1': [seasonal_results['summer_best'][0], seasonal_results['winter_best'][0]],
        'param2': [seasonal_results['summer_best'][1], seasonal_results['winter_best'][1]],
        'energy_kWh': [
            seasonal_results['summer_matrix'].loc[seasonal_results['summer_best']].item(),
            seasonal_results['winter_matrix'].loc[seasonal_results['winter_best']].item()
        ]
    }
    pd.DataFrame(seasonal_summary).to_csv("csv/seasonal_optimal_orientations.csv", index=False)

    # Save full matrices separately
    seasonal_results['summer_matrix'].to_csv("csv/summer_energy_matrix.csv")
    seasonal_results['winter_matrix'].to_csv("csv/winter_energy_matrix.csv")
    plot_annual_and_seasons([annual_matrix, seasonal_results['summer_matrix'], seasonal_results['winter_matrix']],
                            ['Annual', 'Summer (03-20 to 09-22)', 'Winter (09-23 to 03-19)'], 'plots/total_seasonal_matrices.png')

    plot_monthly_heatmaps(monthly_map, SIM_PARAM1, SIM_PARAM2, 'plots/monthly_heatmaps.png')

    plot_optimal_orientation_over_time(monthly_results, 'plots/optimal_orientation_over_time.png')

    plot_daily_energy_lines(results, best_param1_annual, best_param2_annual,
                            monthly_results, seasonal_results, 'plots/daily_energy_comparison.png')
    print(f"\n--- Optimal Orientation Results ---")
    yearly_energy_monthly_adjust = 0
    for res in monthly_results:
        print(f"{res['month']} optimum: {SIM_PARAM1_NAME}: {res['param1']}, {SIM_PARAM2_NAME}: {res['param2']}, Total Production: {res['energy_kWh']:.1f} kWh")
        yearly_energy_monthly_adjust += res['energy_kWh']
    print(f"Summer optimum: {SIM_PARAM1_NAME}: {seasonal_summary['param1'][0]}, {SIM_PARAM2_NAME}: {seasonal_summary['param2'][0]}, Total Production: {seasonal_summary['energy_kWh'][0]:.1f} kWh")
    print(f"Winter optimum: {SIM_PARAM1_NAME}: {seasonal_summary['param1'][1]}, {SIM_PARAM2_NAME}: {seasonal_summary['param2'][1]}, Total Production: {seasonal_summary['energy_kWh'][1]:.1f} kWh")
    print(f"Yearly optimum: {SIM_PARAM1_NAME}: {best_param1_annual}, {SIM_PARAM2_NAME}: {best_param2_annual}, Total Production: {annual_matrix.max().max():.1f} kWh")
    print(f"Total energy when not adjusting orientation: {annual_matrix.max().max():.1f} kWh")
    print(f"Total energy when adjusting at equinoxes: {sum(seasonal_summary['energy_kWh']):.1f} kWh")
    print(f"Total energy when adjusting monthly: {yearly_energy_monthly_adjust:.1f} kWh")
    return best_param1_annual, best_param2_annual

def build_energy_matrices(results, param1, param2):
    annual_matrix = pd.DataFrame(index=param1, columns=param2, dtype=float)
    monthly_map = {}

    for param1, param2, daily_series in results:
        annual_matrix.at[param1, param2] = daily_series.sum()
        monthly_series = daily_series.resample('MS').sum()
        monthly_map[(param1, param2)] = monthly_series

    return annual_matrix, monthly_map

def find_optimal_orientation(matrix):
    idx = np.unravel_index(np.argmax(matrix.values), matrix.shape)
    return matrix.index[idx[0]], matrix.columns[idx[1]]

def compute_monthly_optimal_orientations(monthly_map):
    months = sorted({date for s in monthly_map.values() for date in s.index})
    results = []
    for month in months:
        best_val = -np.inf
        best_param1 = None
        best_param2 = None
        for (param1, param2), series in monthly_map.items():
            val = series.get(month, 0)
            if val > best_val:
                best_val = val
                best_param1, best_param2 = param1, param2
        results.append({'month': month.strftime('%Y-%m'), 'param1': best_param1, 'param2': best_param2, 'energy_kWh': best_val})
    return results

def compute_seasonal_optimal_orientations(results, solpos):
    equinox_spring = pd.Timestamp(f"{solpos.index[0].year}-03-20", tz=solpos.index.tz)
    equinox_fall = pd.Timestamp(f"{solpos.index[0].year}-09-22", tz=solpos.index.tz)

    summer_days = solpos[(solpos.index >= equinox_spring) & (solpos.index < equinox_fall)].index.normalize().unique()
    winter_days = solpos[(solpos.index < equinox_spring) | (solpos.index >= equinox_fall)].index.normalize().unique()

    def seasonal_matrix(daylist):
        matrix = pd.DataFrame(index=SIM_PARAM1, columns=SIM_PARAM2, dtype=float)
        for param1, param2, daily_series in results:
            values = daily_series[daily_series.index.normalize().isin(daylist)]
            matrix.at[param1, param2] = values.sum()
        return matrix

    summer_matrix = seasonal_matrix(summer_days)
    winter_matrix = seasonal_matrix(winter_days)

    summer_best = find_optimal_orientation(summer_matrix)
    winter_best = find_optimal_orientation(winter_matrix)

    return {
        'summer_matrix': summer_matrix,
        'winter_matrix': winter_matrix,
        'summer_best': summer_best,
        'winter_best': winter_best
    }

def plot_annual_and_seasons(matrices, titles, filename):
    fig, axs = plt.subplots(1, len(matrices), figsize=(18, 6))
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        c = axs[i].contourf(matrix.index, matrix.columns, matrix.values.T, levels=20, cmap='viridis')
        axs[i].set_title(title)
        axs[i].set_xlabel(SIM_PARAM1_NAME)
        axs[i].set_ylabel(SIM_PARAM2_NAME)
        fig.colorbar(c, ax=axs[i])
    plt.tight_layout()
    plt.savefig(filename)

def plot_monthly_heatmaps(monthly_map, param1, param2, filename):
    months = sorted({date for s in monthly_map.values() for date in s.index})
    fig, axs = plt.subplots(3, 4, figsize=(16, 10))
    for idx, month in enumerate(months[:12]):
        data = pd.DataFrame(index=param1, columns=param2, dtype=float)
        for (monthly_param1, monthly_param2), series in monthly_map.items():
            data.at[monthly_param1, monthly_param2] = series.get(month, 0)
        ax = axs[idx // 4, idx % 4]
        c = ax.contourf(data.index, data.columns, data.values.T, levels=20, cmap='viridis')
        ax.set_title(month.strftime('%b'))
        ax.set_xlabel(SIM_PARAM1_NAME)
        ax.set_ylabel(SIM_PARAM2_NAME)
        fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.savefig(filename)

def plot_optimal_orientation_over_time(monthly_results, output_file):
    df = pd.DataFrame(monthly_results)
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values('month')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_tilt = 'tab:blue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel(SIM_PARAM1_NAME, color=color_tilt)
    ax1.plot(df['month'], df['param1'], color=color_tilt, marker='o', label=SIM_PARAM1_NAME)
    ax1.tick_params(axis='y', labelcolor=color_tilt)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color_azimuth = 'tab:red'
    ax2.set_ylabel(SIM_PARAM2_NAME, color=color_azimuth)
    ax2.plot(df['month'], df['param2'], color=color_azimuth, marker='s', label=SIM_PARAM2_NAME)
    ax2.tick_params(axis='y', labelcolor=color_azimuth)
    plt.title('Monthly Optimal Parameters')
    plt.savefig(output_file)

def plot_daily_energy_lines(results, best_param1_annual, best_param2_annual,
                            monthly_results, seasonal_results, output_file):
    series_map = {(param1, param2): daily_series for param1, param2, daily_series in results}
    fig, ax = plt.subplots(figsize=(14, 7))

    # Annual optimal
    annual_series = series_map[(best_param1_annual, best_param2_annual)].rolling(window=14, center=True).mean()
    annual_series.plot(label="Annual Optimal", color='black', ax=ax)

    # Monthly optimal
    for row in monthly_results:
        param1, param2 = row['param1'], row['param2']
        series = series_map[(param1, param2)].rolling(window=14, center=True).mean()
        series.loc[row['month']].plot(ax=ax, style='--', alpha=0.5, label=f"{row['month']} Optimal")  # optional

    # Seasonal optimal
    for season in ['summer', 'winter']:
        param1, param2 = seasonal_results[f'{season}_best']
        series = series_map[(param1, param2)].rolling(window=14, center=True).mean()
        series.plot(label=f"{season.capitalize()} Optimal", ax=ax)

    ax.set_title("Daily Energy Production With Average Weather (14d moving average)")
    ax.set_ylabel("Energy (kWh)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)

def plot_daily_performance(loc, module_params, inverter_params, temp_model_params,
                           tmy_data, optimal_param1, optimal_param2, day_to_plot, horizon_profiles):
    """
    Plots the predicted AC power output for a specific day using ModelChain,
    considering average weather and clear sky conditions, with horizon masking.
    """
    print(f"\n📊 Plotting daily performance for {day_to_plot} at optimal parameters ({SIM_PARAM1_NAME}: {optimal_param1}, {SIM_PARAM2_NAME}: {optimal_param2})...")

    # Define 5-minute time range for the specified day
    start_time = pd.to_datetime(day_to_plot)
    end_time = start_time + pd.Timedelta(days=1)
    times_5min = pd.date_range(start=start_time, end=end_time, freq=f'{SIM_GRANULARITY_MIN}min', inclusive='left', tz=TIME_ZONE)

    # Get solar position for 5-minute intervals
    solpos_5min = get_solarposition(times_5min, loc.latitude, loc.longitude)

    # --- Prepare weather data for the specified day (5-minute resolution) ---
    # Interpolate hourly TMY data to 5-minute intervals for the plot day
    day_tmy_data_day = tmy_data.loc[day_to_plot] # Select relevant day

    # Get clear sky data for the same 5-minute intervals
    clearsky_data_5min = loc.get_clearsky(times_5min)

    # Apply horizon masking to both TMY and clear sky data for the day
    masked_tmy_weather = list(map(lambda x: apply_horizon_mask_to_weather(day_tmy_data_day, solpos_5min, x), horizon_profiles))
    masked_clearsky_weather = list(map(lambda x: apply_horizon_mask_to_weather(clearsky_data_5min, solpos_5min, x), horizon_profiles))

    # --- Setup PVSystem for the optimal orientation ---
    panel1_tilt = fill_params(PANEL_ORIENTATIONS[0]['tilt'], optimal_param1, optimal_param2)
    panel2_tilt = fill_params(PANEL_ORIENTATIONS[1]['tilt'], optimal_param1, optimal_param2)
    panel1_azimuth = fill_params(PANEL_ORIENTATIONS[0]['azimuth'], optimal_param1, optimal_param2)
    panel2_azimuth = fill_params(PANEL_ORIENTATIONS[1]['azimuth'], optimal_param1, optimal_param2)
    array1 = pvsystem.Array(
        mount=pvsystem.FixedMount(surface_tilt=panel1_tilt, surface_azimuth=panel1_azimuth),
        module_parameters=module_params,
        temperature_model_parameters=temp_model_params,
        modules_per_string=1,
        strings=1,
        name='MPPT1 Array'
    )
    array2 = pvsystem.Array(
        mount=pvsystem.FixedMount(surface_tilt=panel2_tilt, surface_azimuth=panel2_azimuth),
        module_parameters=module_params,
        temperature_model_parameters=temp_model_params,
        modules_per_string=1,
        strings=1,
        name='MPPT2 Array'
    )
    arrays = [array1, array2]
    system = pvsystem.PVSystem(arrays=arrays, inverter_parameters=inverter_params, albedo=0.2)

    # --- Run ModelChain for TMY data (average weather) ---
    mc_tmy = modelchain.ModelChain(system, loc,
                                    clearsky_model='ineichen',
                                    temperature_model='sapm',
                                    dc_model='cec',
                                    ac_model='sandia',
                                    aoi_model='physical')
    mc_tmy.run_model(weather=masked_tmy_weather)
    power_tmy = mc_tmy.results.ac

    # --- Run ModelChain for Clear Sky data ---
    mc_clearsky = modelchain.ModelChain(system, loc,
                                         clearsky_model='ineichen',
                                         temperature_model='sapm',
                                         dc_model='cec',
                                         ac_model='sandia',
                                         aoi_model='physical')
    mc_clearsky.run_model(weather=masked_clearsky_weather)
    power_clearsky = mc_clearsky.results.ac

    # Calculate total energy for the day
    # Power is in Watts, times are 5-minute intervals (1/12 of an hour)
    total_energy_tmy_wh = power_tmy.sum() * (SIM_GRANULARITY_MIN / 60)
    total_energy_clearsky_wh = power_clearsky.sum() * (SIM_GRANULARITY_MIN / 60)

    print(f"Total power on {day_to_plot} with average weather: {total_energy_tmy_wh:.2f} Wh")
    print(f"Total power on {day_to_plot} with clear sky: {total_energy_clearsky_wh:.2f} Wh")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(times_5min, power_tmy, label='Power (Average TMY Weather) [W]', color='orange')
    plt.plot(times_5min, power_clearsky, label='Power (Clear Sky) [W]', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.title(f"Predicted AC Output on {day_to_plot} at Yearly Optimal Orientation")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=TIME_ZONE))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/example-day-{day_to_plot}.png")


def main():
    """
    Main function to orchestrate fetching TMY data, finding optimal panel orientation,
    and plotting daily performance.
    """
    np.seterr(all='raise')
    print("--- Starting Solar Panel Simulation ---")

    module = PANEL
    inverter = INVERTER

    # Define temperature model parameters for open rack glass-glass modules.
    temp_model_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][PANEL_TEMPERATURE_MODEL]

    # Create a Location object for Munich.
    loc = location.Location(LATITUDE, LONGITUDE, tz=TIME_ZONE)

    # 2. Fetch TMY data from PVGIS (hourly data for annual simulation)
    print(f"📥 Fetching TMY data from PVGIS...")
    # Using usehorizon=False as we'll apply our custom horizon mask
    tmy_data_hourly, meta = get_pvgis_tmy(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        outputformat='json',
        usehorizon=False,
        coerce_year=2025
    )
    tmy_data = tmy_data_hourly.tz_convert(TIME_ZONE).resample(f'{SIM_GRANULARITY_MIN}min').asfreq().ffill()
    print("TMY data fetched successfully.")

    # Calculate solar position for the entire year's hourly TMY data
    print("🔍 Calculating solar position for annual data...")
    solpos_hourly = get_solarposition(tmy_data.index, LATITUDE, LONGITUDE)

    # 3. Build Horizon Profile
    print("⚙️ Building horizon profile...")
    horizon_profiles = list(map(lambda x: build_horizon_profile(x), HORIZON_PROFILE_DATA))

    # 4. Simulate optimal orientation over a year
    # This function will handle applying the horizon mask internally to the hourly data
    optimal_param1, optimal_param2 = simulate_optimal_orientation(
        loc, module, inverter, temp_model_params,
        tmy_data, solpos_hourly, horizon_profiles
    )

    # 5. Plot daily performance using the optimal angles
    for day in PLOT_DAYS:
        plot_daily_performance(
            loc, module, inverter, temp_model_params,
            tmy_data, optimal_param1, optimal_param2, day, horizon_profiles
        )
    plt.show()

if __name__ == "__main__":
    main()
