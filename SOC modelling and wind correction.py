import rasterio
import numpy as np
import xarray as xr
import scipy.stats as stats
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.transform import from_bounds
import os

# Step 1: Load SOC data from Trend.Earth (assume ΔSOC raster, e.g., 'soc_trend.tif')
soc_path = 'data/soc_trend.tif'  # Update path to your Trend.Earth SOC change raster (t C/ha or kg C m²)
with rasterio.open(soc_path) as src_soc:
    soc_data = src_soc.read(1).astype(np.float32)  # Read first band (SOC trend/change)
    soc_transform = src_soc.transform
    soc_profile = src_soc.profile
    bounds = src_soc.bounds

# Mask NoData
soc_data = np.ma.masked_where(soc_data == src_soc.nodata, soc_data)

# Step 2: Load wind data (e.g., ERA5 netCDF for u/v components; download from CDS)
# Assume 'wind_era5.nc' with 'u10' (eastward), 'v10' (northward) wind at 10m
wind_ds = xr.open_dataset('data/wind_era5.nc')  # Update path; average over period if multi-time
u_wind = wind_ds['u10'].mean(dim='time').values  # Eastward component
v_wind = wind_ds['v10'].mean(dim='time').values  # Northward component

# Compute wind speed and direction
wind_speed = np.sqrt(u_wind**2 + v_wind**2)  # Speed (m/s)
wind_dir = (np.arctan2(v_wind, u_wind) * 180 / np.pi + 360) % 360  # Direction (degrees, meteorological convention)

# Resample wind to match SOC raster resolution (assume same CRS; use affine transform if needed)
# For simplicity, assume wind is already aligned or use resampling (e.g., via xarray interp)
wind_speed_resampled = wind_ds['u10'].mean(dim='time').interp(lon=src_soc.bounds.left, lat=src_soc.bounds.top).values  # Placeholder; adjust coords

# Step 3: Align arrays (assume same shape after resampling; pad/crop as needed)
# Example: Crop wind to SOC bounds if larger
min_row, max_row = ...  # Compute indices based on bounds
wind_speed_aligned = wind_speed[min_row:max_row, ...]  # Adjust slicing
soc_aligned = soc_data  # Assume matching

# Mask invalid values
valid_mask = ~np.ma.getmaskarray(soc_aligned) & ~np.isnan(wind_speed_aligned)
soc_valid = soc_aligned[valid_mask]
wind_valid = wind_speed_aligned[valid_mask]

# Step 4: Correlation analysis (Pearson r between SOC change and wind speed)
r, p_value = stats.pearsonr(soc_valid, wind_valid)
print(f"Pearson correlation (SOC vs Wind Speed): r = {r:.3f}, p = {p_value:.3e}")

# For direction: Binned correlation or vector analysis (e.g., by wind quadrants)
# Example: Group by direction bins (N, NE, E, etc.)
dir_bins = np.digitize(wind_dir[valid_mask], bins=[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
for bin_idx in range(1, 9):
    mask_bin = dir_bins == bin_idx
    if np.sum(mask_bin) > 10:  # Min samples
        r_bin, p_bin = stats.pearsonr(soc_valid[mask_bin], wind_valid[mask_bin])
        print(f"Direction bin {bin_idx*45-22.5:.0f}°: r = {r_bin:.3f}, p = {p_bin:.3e}")

# Step 5: Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# SOC map
show(soc_data, transform=soc_transform, ax=ax1, cmap='RdYlBu')
ax1.set_title('SOC Change (Trend.Earth)')

# Wind speed map (overlay)
show(wind_speed_aligned, transform=from_bounds(*bounds, soc_data.shape[1], soc_data.shape[0]), ax=ax2, cmap='viridis')
ax2.set_title('Mean Wind Speed (m/s)')

plt.tight_layout()
plt.savefig('output/soc_wind_correlation.png', dpi=300)
plt.show()

# Save correlation stats
stats_df = pd.DataFrame({'r': [r], 'p_value': [p_value]})
stats_df.to_csv('output/soc_wind_correlation.csv', index=False)
print("Maps and stats saved in output/")
