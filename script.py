import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
EXCEL_FILE = "marine.xlsx"      # Your input file
SHEET_NAME = 0                 # Sheet index or name
LAT_COL = 0                    # Column index for latitude (0-based)
LON_COL = 1                    # Column index for longitude (0-based)
OUTPUT_FILE = "output.xlsx"    # Output file
ONSHORE_BUFFER_KM = 3         # Threshold for onshore classification (in km)

def main():
    try:
        logger.info(f"Loading data from {EXCEL_FILE}...")
        
        # --- Load Data ---
        df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
        logger.info(f"Loaded {len(df)} rows")
        
        # Ensure we have enough columns
        if len(df.columns) <= max(LAT_COL, LON_COL):
            raise ValueError(f"Excel file must have at least {max(LAT_COL, LON_COL) + 1} columns")
        
        # Get lat, lon columns
        latitudes = df.iloc[:, LAT_COL]
        longitudes = df.iloc[:, LON_COL]
        
        # Check for invalid coordinates
        invalid_coords = pd.isna(latitudes) | pd.isna(longitudes)
        if invalid_coords.any():
            logger.warning(f"Found {invalid_coords.sum()} rows with invalid coordinates")
        
        logger.info("Creating point geometries...")
        geometry = []
        for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
            if pd.isna(lat) or pd.isna(lon):
                geometry.append(None)
            else:
                try:
                    # Validate coordinate ranges
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        logger.warning(f"Row {i+1}: Invalid coordinate range (lat: {lat}, lon: {lon})")
                        geometry.append(None)
                    else:
                        geometry.append(Point(lon, lat))
                except Exception as e:
                    logger.error(f"Row {i+1}: Error creating point geometry: {e}")
                    geometry.append(None)
        
        # Create GeoDataFrame
        gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Remove rows with invalid geometries for processing
        valid_mask = gdf_points.geometry.notna()
        gdf_valid = gdf_points[valid_mask].copy()
        
        if len(gdf_valid) == 0:
            raise ValueError("No valid coordinates found in the data")
        
        logger.info(f"Processing {len(gdf_valid)} valid coordinates...")
        
        # --- Load Physical Features ---
        logger.info("Loading Natural Earth data from local files...")
        
        # Common shapefiles in the 10m physical dataset
        possible_files = [
            "ne_10m_land.shp",              # Land polygons (most likely what we want)
            "ne_10m_coastline.shp",         # Coastline features
            "ne_10m_ocean.shp",             # Ocean polygons
            "ne_10m_geography_regions_polys.shp"  # Geographic regions
        ]
        
        physical_data = None
        
        # Try to find and load the most appropriate file
        for filename in possible_files:
            if os.path.exists(filename):
                logger.info(f"Loading from local file: {filename}")
                physical_data = gpd.read_file(filename)
                break
        
        if physical_data is None:
            raise FileNotFoundError("No Natural Earth shapefiles found. Please ensure you have the required files in the current directory.")
        
        logger.info("Natural Earth data loaded successfully from local file")
        
        # Filter for land features if needed
        logger.info(f"Physical data loaded with {len(physical_data)} features")
        
        # If there's a 'featurecla' or similar column, filter for land features
        if 'featurecla' in physical_data.columns:
            land_features = physical_data[physical_data['featurecla'].isin(['Land', 'land', 'LAND'])]
            if len(land_features) > 0:
                physical_data = land_features
                logger.info(f"Filtered to {len(physical_data)} land features")
        
        # Use the physical data as land polygons
        world = physical_data
        
        # Project to metric coordinate system for distance calculations (Web Mercator)
        logger.info("Projecting to metric coordinate system...")
        gdf_points_proj = gdf_valid.to_crs(epsg=3857)
        world_proj = world.to_crs(epsg=3857)
        
        # Create coastline from physical boundaries
        coastline_proj = world_proj.boundary
        
        # If coastline_proj is a GeoSeries of MultiLineStrings, we need to handle it properly
        if hasattr(coastline_proj, 'unary_union'):
            coastline_unified = coastline_proj.unary_union
        else:
            coastline_unified = coastline_proj
        
        # --- Classification Function ---
        def classify_point(point_proj, point_original):
            try:
                # Check if point is on land (intersects with any land polygon)
                is_land = world_proj.contains(point_proj).any()
                
                if not is_land:
                    return "Offshore"
                else:
                    # Calculate distance to coastline in kilometers
                    if hasattr(coastline_proj, 'distance'):
                        distances = coastline_proj.distance(point_proj)
                        min_distance_km = distances.min() / 1000.0
                    else:
                        # Handle case where coastline_proj is a single geometry
                        distance = point_proj.distance(coastline_unified)
                        min_distance_km = distance / 1000.0
                    
                    if min_distance_km <= ONSHORE_BUFFER_KM:
                        return "Onshore"
                    else:
                        return "Inland"
                        
            except Exception as e:
                logger.error(f"Error classifying point {point_original}: {e}")
                return "Error"
        
        # --- Apply Classification ---
        logger.info("Classifying points...")
        classifications = []
        
        for i, (idx, row) in enumerate(gdf_valid.iterrows()):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(gdf_valid)} points...")
            
            point_proj = gdf_points_proj.loc[idx, 'geometry']
            point_original = row.geometry
            classification = classify_point(point_proj, point_original)
            classifications.append(classification)
        
        # Add classifications to the valid dataframe
        gdf_valid['Shore_Status'] = classifications
        
        # --- Prepare Final Output ---
        # Start with original dataframe and add classification column
        final_df = df.copy()
        final_df['Shore_Status'] = 'Unknown'  # Default for invalid coordinates
        
        # Update with actual classifications for valid coordinates
        final_df.loc[valid_mask, 'Shore_Status'] = classifications
        
        # --- Save Output ---
        logger.info(f"Saving results to {OUTPUT_FILE}...")
        final_df.to_excel(OUTPUT_FILE, index=False)
        
        # Print summary
        status_counts = final_df['Shore_Status'].value_counts()
        logger.info("Classification Summary:")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
        
        logger.info(f"Done! Output saved to {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
        print(f"\nProcessing complete! Check {OUTPUT_FILE} for results.")
        print(f"\nClassification categories:")
        print(f"- Offshore: Points over water")
        print(f"- Onshore: Points on land within {ONSHORE_BUFFER_KM}km of coast")
        print(f"- Inland: Points on land more than {ONSHORE_BUFFER_KM}km from coast")
        print(f"- Unknown: Invalid coordinates")
        print(f"- Error: Processing errors")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Install required packages: pip install geopandas pandas openpyxl")
        print("2. Your Excel file exists and has the correct path")
        print("3. Latitude and longitude columns are in the correct positions")
        print("4. Natural Earth shapefiles are in the current directory")