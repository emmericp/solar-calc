import numpy as np
from horizons import *

# Example config, adjust for your setup

# Location
LATITUDE  = 48.15
LONGITUDE = 11.52
TIME_ZONE = 'Europe/Berlin'

# Simulation parameters
SIM_PARAM1_NAME = "Azimuth"
SIM_PARAM2_NAME = "Tilt"
SIM_PARAM1 = np.arange(0, 360, 10)
SIM_PARAM2 = np.arange(0, 91, 5)
SIM_GRANULARITY_MIN = 30
SIM_WORKERS = None # Defaults to number of CPU cores

PANEL_PARAMS = [
	{
		'azimuth': 'PARAM1', # Use PARAM1/2 to refer to variables above, or use numbers for panels you are not optimizing for
		'tilt': 'PARAM2',
		'modules_per_string': 1,
		'strings': 1,
    },
	{
        'azimuth': 'PARAM1',
		'tilt': 'PARAM2',
		'modules_per_string': 1,
		'strings': 1,
    }
]

# Plot given days as examples after full simulation, useful to compare them against real data
PLOT_DAYS = [
	'2025-06-21',
	'2025-12-21'
]

# Horizon profile data: (azimuth, elevation) pairs in degrees
# This data describes obstructions (e.g., mountains, trees) around the site.
# One entry per panel
HORIZON_PROFILE_DATA = [
    HORIZON_BALCONY_WEST,
	HORIZON_BALCONY_WEST,
]


# See https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Modules.csv
# A 460W panel, the exact stats don't matter too much, you can just adjust STC
PANEL = {
    'Technology': 'Mono-c-Si',
    'STC': 460.0, # Probably enough to just tune this one to your value
    'PTC': 440.9 * 460 / 465.36, # Adjusted proportally to STC
    'A_c': 1.998,
    'Length': 1.762,
    'Width': 1.134,
    'N_s': 108,
    'I_sc_ref': 14.35,
    'V_oc_ref': 39.37,
    'I_mp_ref': 13.59,
    'V_mp_ref': 33.85,
    'alpha_sc': 0.0053172,
    'beta_oc': -0.097565,
    'T_NOCT': 43.6,
    'a_ref': 1.40336,
    'I_L_ref': 14.7849,
    'I_o_ref': 8.59478e-12,
    'R_s': 0.102322,
    'R_sh_ref': 101.571,
    'Adjust': 5.3949,
    'gamma_pmp': -0.308,
    'BIPV': 'N',
}
# See pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS
PANEL_TEMPERATURE_MODEL = "open_rack_glass_glass"

# See https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Inverters.csv
# A slightly customized model the 800W micro inverter I have, it should be good enough
INVERTER = {
    'Vac': 230,
    # No clue, but startup voltage is 20V. I see it reporting data at values as low as 1-2W DC input power
    # Output power isn't reported with fine enough granularity, 1W is a fine guess.
    'Pso': 1,
    'Paco': 800, # Artificial limit
    # Peak efficiency is given as 96%, CEC efficiency as 95.5%, assuming it achieves this at peak (which is probably at ~87% it's real capacity)
    # My real setup just manages ~450W out, but there it's at 94.8% efficiency, so let's say 95% efficient
    'Pdco': 800 / 0.95,
    # No clue, but overall range is 25-55V for my model, and the one I'm adapting here has 22-55V with 35V for VDCO, so just using that
    'Vdco': 35.0,
    # No clue, reference model is probably similar enough
    'C0': -2.36577e-05,
    'C1': -6.50811e-05,
    'C2': 0.00544217,
    'C3': -0.00640233,
    'Pnt': 0.05,
    'Vdcmax': 60.0,
    'Idcmax': 27,
    'Mppt_low': 25.0,
    'Mppt_high': 55.0,
    'CEC_hybrid': 'N'
}
