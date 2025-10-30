import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from skimage import exposure  # For any image enhancement if needed

# Suppress runtime warnings (from your original output)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create output directory
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Load shapefile (update path as needed)
shapefile_path = "data/tONK/Export_Output_3.shp"  # Relative to repo
gdf = gpd.read_file(shapefile_path)
gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.exterior if geom.geom_type == "Polygon" else None)
gdf = gdf.dropna(subset=["geometry"])

# Load raster file (update path as needed)
raster_path = "data/trends_earth_data/datasets/a3aa9518-f745-4ae5-b553-4861c90af299/none_download-data_shape-lon75.721lat26.121-buffer-10.000_Int16.tif"
with rasterio.open(raster_path) as src:
    band_1992 = src.read(1)
    band_2022 = src.read(31)
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Define land cover classes and colors
land_cover_classes = {
    10: "CropLand Rainfed",
    20: "CropLand Irrigated",
    30: "Vegetation",
    80: "Tree Cover",
    150: "Sparse Vegetation",
    190: "Urban Area",
    200: "Bare Areas",
    210: "Water Bodies",
}
land_cover_colors = {
    10: (255, 180, 0),
    20: (150, 100, 0),
    30: (0, 255, 0),
    80: (0, 160, 0),
    150: (255, 245, 215),
    190: (195, 20, 0),
    200: (255, 255, 255),
    210: (0, 70, 200),
}

# Calculate pixel area (~0.0009 km² per pixel at ~10m res for accuracy)
pixel_area_km2 = (src.res[0] / 1000) * (src.res[1] / 1000)  # Dynamic based on raster resolution

# Compute transformation statistics
transformation_changes = {}
label_changes = {}
for old_class, old_label in land_cover_classes.items():
    for new_class, new_label in land_cover_classes.items():
        change_area = np.sum((band_1992 == old_class) & (band_2022 == new_class)) * pixel_area_km2
        if change_area > 0 and old_class != new_class:
            transformation_changes[(old_class, new_class)] = change_area
            label_changes[(old_class, new_class)] = f"{old_label} → {new_label}"

# Save transformation statistics
transformation_df = pd.DataFrame(list(transformation_changes.items()), columns=["Transition", "Area (km²)"])
transformation_df["Transition"] = transformation_df["Transition"].apply(lambda x: label_changes[x])
transformation_csv_path = os.path.join(output_folder, "land_cover_transformation.csv")
transformation_df.to_csv(transformation_csv_path, index=False)

# Generate transformation map
plt.figure(figsize=(10, 6))
change_classes = np.zeros_like(band_1992, dtype=int)
for i, (old_class, new_class) in enumerate(transformation_changes.keys()):
    change_classes[(band_1992 == old_class) & (band_2022 == new_class)] = i + 1

plt.imshow(change_classes, cmap="tab20", extent=extent, origin="upper")
plt.colorbar(label="Land Cover Transition")
plt.title("Land Cover Transformation Map (1992-2022)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
gdf.plot(ax=plt.gca(), facecolor="none", edgecolor="black", linewidth=1)  # Overlay shapefile

# Add legend with specific transformations
legend_patches = [mpatches.Patch(color=plt.cm.tab20(i / len(transformation_changes)), label=label_changes[key]) 
                  for i, key in enumerate(transformation_changes.keys())]
plt.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.25, 1.0))

transformation_map_path = os.path.join(output_folder, "land_cover_transformation_map.png")
plt.savefig(transformation_map_path, dpi=300)
plt.close()

print(f"Transformation statistics and map saved in {output_folder}")
