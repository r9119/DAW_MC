import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import trim_mean


# =============================================================================
# City Population Data (for Population Weighted Mean)
# =============================================================================

# Populations of 5 major Spanish metropolitan areas (Source: ESPON 2018)
CITY_POPULATIONS = {
    "Madrid": 6155116,
    "Barcelona": 5179243,
    "Valencia": 1645342,
    "Seville": 1305342,
    "Bilbao": 987000
}

# Population of Spain
SPAIN_POPULATION = 48590000

# Calculate city population factors (proportion of Spain's population)
CITY_FACTORS = {city: pop / SPAIN_POPULATION for city, pop in CITY_POPULATIONS.items()}

# Combined population of the 5 cities
COMBINED_CITY_POPULATION = sum(CITY_POPULATIONS.values())
COMBINED_CITY_PROPORTION = COMBINED_CITY_POPULATION / SPAIN_POPULATION


# =============================================================================
# Helper Functions
# =============================================================================

def huber_mean(data, delta=1.35):
    """
    Calculate the Huber M-estimator mean for a given dataset.
    Uses iteratively reweighted least squares to compute a robust mean.
    
    Parameters:
    -----------
    data : array-like
        The data to compute the robust mean for
    delta : float
        The Huber tuning parameter (default 1.35 for ~95% efficiency)
    
    Returns:
    --------
    float : The Huber M-estimator mean
    """
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return np.nan
    if len(data) == 1:
        return data[0]
    
    try:
        loc, scale = stats.mstats.huber_mean_scale(data, k=delta)
        return loc
    except:
        return np.median(data)


def trimmed_mean(data, proportion=0.2):
    """
    Calculate the trimmed mean by removing a proportion of extreme values.
    
    For 5 cities with proportion=0.2, this removes the highest and lowest
    values (20% from each side) and averages the remaining 3.
    
    Parameters:
    -----------
    data : array-like
        The data to compute the trimmed mean for
    proportion : float
        Proportion to cut from each end (default 0.2 = 20% each side)
    
    Returns:
    --------
    float : The trimmed mean
    """
    data = np.array(data).flatten()
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return np.nan
    if len(data) < 3:
        return np.mean(data)
    
    try:
        return trim_mean(data, proportiontocut=proportion)
    except:
        return np.mean(data)


def population_weighted_mean(data, city_names=None):
    """
    Calculate the population-weighted mean across cities.
    
    Weights each city's value by its population proportion.
    
    Parameters:
    -----------
    data : array-like
        The data values (one per city)
    city_names : array-like, optional
        City names corresponding to data values
    
    Returns:
    --------
    float : The population-weighted mean
    """
    data = np.array(data).flatten()
    
    if city_names is None:
        # If no city names, use equal weights
        return np.nanmean(data)
    
    city_names = np.array(city_names).flatten()
    
    # Get weights for each city
    weights = np.array([CITY_FACTORS.get(str(city).strip(), 0) for city in city_names])
    
    # Handle NaN values
    valid_mask = ~np.isnan(data) & (weights > 0)
    if not np.any(valid_mask):
        return np.nan
    
    valid_data = data[valid_mask]
    valid_weights = weights[valid_mask]
    
    # Normalize weights to sum to 1
    valid_weights = valid_weights / valid_weights.sum()
    
    return np.sum(valid_data * valid_weights)


def circular_mean(angles_deg):
    """
    Calculate the circular (angular) mean for wind direction data.
    
    Parameters:
    -----------
    angles_deg : array-like
        Wind direction angles in degrees (0-360)
    
    Returns:
    --------
    float : The circular mean in degrees (0-360)
    """
    angles_deg = np.array(angles_deg).flatten()
    angles_deg = angles_deg[~np.isnan(angles_deg)]
    
    if len(angles_deg) == 0:
        return np.nan
    
    angles_rad = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)
    mean_angle_deg = np.rad2deg(mean_angle_rad)
    
    if mean_angle_deg < 0:
        mean_angle_deg += 360
    
    return mean_angle_deg


def create_lag_features(df, columns, lags):
    """
    Create lag features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    columns : list
        Columns to create lag features for
    lags : list of int
        Lag periods in hours
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added lag features
    """
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df, columns, windows):
    """
    Create rolling aggregate features (min, max, mean) for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    columns : list
        Columns to create rolling features for
    windows : list of int
        Rolling window sizes in hours
    
    Returns:
    --------
    pd.DataFrame : DataFrame with added rolling features
    """
    df_rolled = df.copy()
    
    for col in columns:
        for window in windows:
            df_rolled[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window).min()
            df_rolled[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window).max()
            df_rolled[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window).mean()
    
    return df_rolled


def join_energy_weather(energy_df, weather_df, how='inner'):
    """
    Join energy and weather dataframes on their datetime index.
    
    Parameters:
    -----------
    energy_df : pd.DataFrame
        Energy consumption dataframe with datetime index
    weather_df : pd.DataFrame
        Aggregated weather dataframe with datetime index
    how : str
        Join type ('inner', 'left', 'right', 'outer')
    
    Returns:
    --------
    pd.DataFrame : Combined dataframe
    """
    energy_df.index = pd.to_datetime(energy_df.index)
    weather_df.index = pd.to_datetime(weather_df.index)
    
    combined_df = energy_df.join(weather_df, how=how, lsuffix='_energy', rsuffix='_weather')
    
    return combined_df


def check_timestamp_alignment(energy_df, weather_df, city_col='city_name', time_col='dt_iso'):
    """
    Check timestamp alignment between energy and weather datasets.
    
    Verifies that timestamps in the energy dataset match those in the weather
    dataset for each city. Reports matching, missing, and extra timestamps.
    
    Parameters:
    -----------
    energy_df : pd.DataFrame
        Energy dataframe with datetime index
    weather_df : pd.DataFrame
        Weather dataframe (can be indexed or have time column)
    city_col : str
        Column name for city in weather data (default 'city_name')
    time_col : str
        Column name for time in weather data if not indexed (default 'dt_iso')
    
    Returns:
    --------
    dict : Alignment statistics per city with keys:
        - 'overlap': number of matching timestamps
        - 'missing_in_weather': timestamps in energy but not weather
        - 'extra_in_weather': timestamps in weather but not energy
        - 'alignment_pct': percentage of energy timestamps covered
    """
    # Get energy timestamps
    energy_timestamps = set(energy_df.index)
    
    # Get weather dataframe with proper structure
    if time_col in weather_df.columns:
        weather_reset = weather_df.copy()
    else:
        weather_reset = weather_df.reset_index()
        if weather_df.index.name:
            time_col = weather_df.index.name
    
    # Ensure time column is datetime
    weather_reset[time_col] = pd.to_datetime(weather_reset[time_col])
    
    # Get timestamps per city
    weather_timestamps_per_city = weather_reset.groupby(city_col)[time_col].apply(set)
    
    alignment_stats = {}
    
    for city in weather_reset[city_col].unique():
        city_timestamps = weather_timestamps_per_city[city]
        overlap = len(energy_timestamps & city_timestamps)
        missing_in_weather = len(energy_timestamps - city_timestamps)
        extra_in_weather = len(city_timestamps - energy_timestamps)
        alignment_pct = (overlap / len(energy_timestamps) * 100) if len(energy_timestamps) > 0 else 0
        
        alignment_stats[city] = {
            'overlap': overlap,
            'missing_in_weather': missing_in_weather,
            'extra_in_weather': extra_in_weather,
            'alignment_pct': alignment_pct
        }
    
    return alignment_stats


def get_aggregation_justification():
    """
    Return statistics justifying the use of 5 Spanish cities for weather aggregation.
    
    Returns:
    --------
    dict : Statistics about city populations and coverage
    """
    total_city_pop = sum(CITY_POPULATIONS.values())
    coverage_pct = (total_city_pop / SPAIN_POPULATION) * 100
    
    return {
        'cities': list(CITY_POPULATIONS.keys()),
        'city_populations': CITY_POPULATIONS,
        'total_city_population': total_city_pop,
        'spain_population': SPAIN_POPULATION,
        'coverage_percentage': coverage_pct,
        'city_factors': CITY_FACTORS
    }


# =============================================================================
# Data Wrangling Pipeline Class
# =============================================================================

class DataWranglingPipeline:
    """
    Generalized data wrangling pipeline for energy and weather data.
    
    Pipeline Steps:
    1. Load data (LE1: Importieren)
    2. Clean data (LE2: Bereinigen)
    3. Transform/Aggregate weather data (LE3: Transformieren)
    4. Join datasets (LE4: Verknüpfen)
    5. Feature engineering (optional)
    
    Supported Aggregation Methods:
    - 'huber': Huber M-estimator (robust to outliers)
    - 'trimmed': Trimmed mean (removes 20% from each end)
    - 'population_weighted': Population-weighted mean
    - 'mean': Simple arithmetic mean
    - 'median': Median
    """
    
    def __init__(self, 
                 aggregation_method='huber', 
                 huber_delta=1.35,
                 trimmed_proportion=0.2,
                 weather_numeric_cols=None,
                 weather_binary_cols=None,
                 circular_cols=None,
                 key_weather_cols=None,
                 create_lag_features=True,
                 create_rolling_features=True,
                 lag_periods=None,
                 rolling_windows=None,
                 target_col='total load actual',
                 energy_time_col='time',
                 weather_time_col='dt_iso'):
        """
        Initialize the pipeline with configuration.
        
        Parameters:
        -----------
        aggregation_method : str
            Method for aggregating weather data:
            'huber', 'trimmed', 'population_weighted', 'mean', 'median'
        huber_delta : float
            Tuning parameter for Huber estimator (default 1.35)
        trimmed_proportion : float
            Proportion to trim from each end for trimmed mean (default 0.2)
        weather_numeric_cols : list, optional
            Numeric columns to aggregate.
        weather_binary_cols : list, optional
            Binary columns to aggregate using max (OR logic).
        circular_cols : list, optional
            Circular columns (e.g., wind_deg) for circular mean.
        key_weather_cols : list, optional
            Key weather columns for lag/rolling features.
        create_lag_features : bool
            Whether to create lag features (default True)
        create_rolling_features : bool
            Whether to create rolling features (default True)
        lag_periods : list, optional
            Lag periods in hours for feature engineering.
        rolling_windows : list, optional
            Rolling window sizes in hours for feature engineering.
        target_col : str
            Target column name in energy data.
        energy_time_col : str
            Time column name in energy data.
        weather_time_col : str
            Time column name in weather data.
        """
        self.aggregation_method = aggregation_method
        self.huber_delta = huber_delta
        self.trimmed_proportion = trimmed_proportion
        self.target_col = target_col
        self.energy_time_col = energy_time_col
        self.weather_time_col = weather_time_col
        
        # Feature engineering flags
        self._create_lag_features = create_lag_features
        self._create_rolling_features = create_rolling_features
        
        # Default numeric columns for aggregation
        self.weather_numeric_cols = weather_numeric_cols or [
            'temp', 'temp_min', 'temp_max', 'pressure', 
            'humidity', 'wind_speed',
            'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all'
        ]
        
        # Binary columns (aggregated with max = OR logic)
        self.weather_binary_cols = weather_binary_cols or [
            'has_rain', 'has_drizzle', 'has_mist', 'has_fog',
            'has_thunderstorm', 'has_snow', 'has_clouds', 
            'has_clear', 'has_haze', 'has_dust'
        ]
        
        # Circular columns
        self.circular_cols = circular_cols or ['wind_deg']
        
        # Key columns for lag/rolling features
        self.key_weather_cols = key_weather_cols or ['temp', 'humidity', 'pressure', 'wind_speed']
        
        # Feature engineering parameters
        self.lag_periods = lag_periods if lag_periods is not None else [1, 24, 168]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [24, 168]
        
    def load_data(self, energy_path, weather_path):
        """Step 1: Load raw data from CSV files."""
        print("Step 1: Loading data...")
        energy_df = pd.read_csv(energy_path)
        weather_df = pd.read_csv(weather_path)
        print(f"  Energy data: {energy_df.shape}")
        print(f"  Weather data: {weather_df.shape}")
        return energy_df, weather_df
    
    def clean_data(self, energy_df, weather_df):
        """Step 2: Clean and preprocess data."""
        print("Step 2: Cleaning data...")
        
        energy_df = energy_df.copy()
        weather_df = weather_df.copy()
        
        # Convert datetime columns and set as index
        energy_df[self.energy_time_col] = pd.to_datetime(energy_df[self.energy_time_col], utc=True)
        energy_df = energy_df.set_index(self.energy_time_col)
        
        weather_df[self.weather_time_col] = pd.to_datetime(weather_df[self.weather_time_col], utc=True)
        weather_df = weather_df.set_index(self.weather_time_col)
        
        # Clean city names (remove whitespace)
        if 'city_name' in weather_df.columns:
            weather_df['city_name'] = weather_df['city_name'].str.strip()
        
        # Fill missing values in energy data using forecast
        if self.target_col in energy_df.columns:
            forecast_col = self.target_col.replace('actual', 'forecast')
            if forecast_col in energy_df.columns:
                energy_df[self.target_col] = energy_df[self.target_col].fillna(
                    energy_df[forecast_col]
                )
        
        missing_target = energy_df[self.target_col].isna().sum() if self.target_col in energy_df.columns else 'N/A'
        print(f"  Missing values in '{self.target_col}': {missing_target}")
        return energy_df, weather_df
    
    def check_timestamp_alignment(self, energy_df, weather_df):
        """
        Check timestamp alignment between energy and weather datasets.
        
        Verifies that timestamps in the energy dataset match those in the weather
        dataset for each city. Reports matching, missing, and extra timestamps.
        
        Parameters:
        -----------
        energy_df : pd.DataFrame
            Energy dataframe with datetime index
        weather_df : pd.DataFrame
            Weather dataframe with datetime index
        
        Returns:
        --------
        dict : Alignment statistics per city
        """
        print("Step 2b: Checking timestamp alignment...")
        
        alignment_stats = check_timestamp_alignment(
            energy_df, 
            weather_df, 
            city_col='city_name',
            time_col=self.weather_time_col
        )
        
        print("  --- Timestamp alignment per city ---")
        all_aligned = True
        for city, stats in alignment_stats.items():
            status = "✓" if stats['missing_in_weather'] == 0 else "⚠"
            if stats['missing_in_weather'] > 0:
                all_aligned = False
            print(f"  {status} {city}: {stats['overlap']} matching, "
                  f"{stats['missing_in_weather']} missing in weather, "
                  f"{stats['extra_in_weather']} extra in weather "
                  f"({stats['alignment_pct']:.1f}% coverage)")
        
        if all_aligned:
            print("  All timestamps aligned!")
        else:
            print("  Warning: Some timestamps are missing in weather data")
        
        return alignment_stats
    
    def aggregate_weather(self, weather_df):
        """Step 3: Aggregate weather data from multiple cities."""
        print(f"Step 3: Aggregating weather data using '{self.aggregation_method}' method...")
        
        # Filter to available columns
        available_numeric = [col for col in self.weather_numeric_cols if col in weather_df.columns]
        available_binary = [col for col in self.weather_binary_cols if col in weather_df.columns]
        available_circular = [col for col in self.circular_cols if col in weather_df.columns]
        
        aggregated_parts = []
        
        # Aggregate numeric columns based on method
        if available_numeric:
            if self.aggregation_method == 'huber':
                numeric_agg = weather_df.groupby(weather_df.index)[available_numeric].agg(
                    lambda x: huber_mean(x.values, delta=self.huber_delta)
                )
            elif self.aggregation_method == 'trimmed':
                numeric_agg = weather_df.groupby(weather_df.index)[available_numeric].agg(
                    lambda x: trimmed_mean(x.values, proportion=self.trimmed_proportion)
                )
            elif self.aggregation_method == 'population_weighted':
                # For population weighted, we need city names
                def pop_weighted_agg(group):
                    result = {}
                    city_names = weather_df.loc[group.index, 'city_name'].values if 'city_name' in weather_df.columns else None
                    for col in available_numeric:
                        result[col] = population_weighted_mean(group[col].values, city_names)
                    return pd.Series(result)
                
                numeric_agg = weather_df.groupby(weather_df.index).apply(
                    lambda g: pd.Series({
                        col: population_weighted_mean(
                            g[col].values,
                            g['city_name'].values if 'city_name' in g.columns else None
                        ) for col in available_numeric
                    })
                )
            elif self.aggregation_method == 'mean':
                numeric_agg = weather_df.groupby(weather_df.index)[available_numeric].mean()
            elif self.aggregation_method == 'median':
                numeric_agg = weather_df.groupby(weather_df.index)[available_numeric].median()
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            aggregated_parts.append(numeric_agg)
            print(f"  Aggregated {len(available_numeric)} numeric columns")
        
        # Aggregate binary columns using max
        if available_binary:
            binary_agg = weather_df.groupby(weather_df.index)[available_binary].max()
            aggregated_parts.append(binary_agg)
            print(f"  Aggregated {len(available_binary)} binary columns (using max/OR)")
        
        # Aggregate circular columns using circular mean
        if available_circular:
            circular_agg = weather_df.groupby(weather_df.index)[available_circular].agg(
                lambda x: circular_mean(x.values)
            )
            aggregated_parts.append(circular_agg)
            print(f"  Aggregated {len(available_circular)} circular columns")
        
        # Combine all aggregated parts
        if aggregated_parts:
            aggregated = pd.concat(aggregated_parts, axis=1)
        else:
            raise ValueError("No columns available for aggregation!")
        
        print(f"  Aggregated from {weather_df.shape[0]} rows to {aggregated.shape[0]} rows")
        return aggregated
    
    def engineer_features(self, weather_df):
        """Step 4: Create lag and rolling features (if enabled)."""
        print("Step 4: Engineering features...")
        
        available_key_cols = [col for col in self.key_weather_cols if col in weather_df.columns]
        
        df = weather_df.copy()
        
        # Create lag features if enabled
        if self._create_lag_features and self.lag_periods and available_key_cols:
            df = create_lag_features(df, available_key_cols, self.lag_periods)
            print(f"  Created lag features for {len(available_key_cols)} columns with lags {self.lag_periods}")
        else:
            print("  Lag features: disabled")
        
        # Create rolling features for temperature if enabled
        if self._create_rolling_features and self.rolling_windows and 'temp' in df.columns:
            df = create_rolling_features(df, ['temp'], self.rolling_windows)
            print(f"  Created rolling features with windows {self.rolling_windows}")
        else:
            print("  Rolling features: disabled")
        
        new_features = df.shape[1] - weather_df.shape[1]
        print(f"  Added {new_features} new features")
        return df
    
    def join_datasets(self, energy_df, weather_df):
        """Step 5: Join energy and weather data."""
        print("Step 5: Joining datasets...")
        
        combined = join_energy_weather(energy_df, weather_df, how='inner')
        print(f"  Combined shape: {combined.shape}")
        print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
        return combined
    
    def run(self, energy_path, weather_path, drop_na_subset=None):
        """
        Execute the complete pipeline.
        
        Parameters:
        -----------
        energy_path : str
            Path to energy CSV file
        weather_path : str
            Path to weather CSV file
        drop_na_subset : list, optional
            Columns to check for NaN when dropping rows.
            
        Returns:
        --------
        pd.DataFrame : Fully processed and combined dataset
        """
        print("=" * 60)
        print("DATA WRANGLING PIPELINE")
        print(f"Aggregation Method: {self.aggregation_method}")
        print(f"Lag Features: {'enabled' if self._create_lag_features else 'disabled'}")
        print(f"Rolling Features: {'enabled' if self._create_rolling_features else 'disabled'}")
        print("=" * 60)
        
        # Step 1: Load
        energy_df, weather_df = self.load_data(energy_path, weather_path)
        
        # Step 2: Clean
        energy_df, weather_df = self.clean_data(energy_df, weather_df)
        
        # Step 2b: Check timestamp alignment
        self.check_timestamp_alignment(energy_df, weather_df)
        
        # Step 3: Aggregate weather
        weather_aggregated = self.aggregate_weather(weather_df)
        
        # Step 4: Feature engineering
        weather_features = self.engineer_features(weather_aggregated)
        
        # Step 5: Join
        combined = self.join_datasets(energy_df, weather_features)
        
        # Handle NaN values
        initial_rows = len(combined)
        
        # Skip rows at the beginning due to lag/rolling features
        rows_to_skip = 0
        if self._create_lag_features and self.lag_periods:
            rows_to_skip = max(rows_to_skip, max(self.lag_periods))
        if self._create_rolling_features and self.rolling_windows:
            rows_to_skip = max(rows_to_skip, max(self.rolling_windows))
        
        if rows_to_skip > 0:
            combined = combined.iloc[rows_to_skip:]
        
        # Drop remaining NaN rows
        if drop_na_subset:
            combined = combined.dropna(subset=drop_na_subset)
        else:
            combined = combined.dropna(subset=[self.target_col])
        
        dropped_rows = initial_rows - len(combined)
        
        print("=" * 60)
        print(f"Pipeline complete! Final dataset: {combined.shape}")
        print(f"Dropped {dropped_rows} rows (first {rows_to_skip} due to lag/rolling features)")
        print("=" * 60)
        
        return combined
    
    def get_config(self):
        """Return current pipeline configuration as a dictionary."""
        return {
            'aggregation_method': self.aggregation_method,
            'huber_delta': self.huber_delta,
            'trimmed_proportion': self.trimmed_proportion,
            'weather_numeric_cols': self.weather_numeric_cols,
            'weather_binary_cols': self.weather_binary_cols,
            'circular_cols': self.circular_cols,
            'key_weather_cols': self.key_weather_cols,
            'create_lag_features': self._create_lag_features,
            'create_rolling_features': self._create_rolling_features,
            'lag_periods': self.lag_periods,
            'rolling_windows': self.rolling_windows,
            'target_col': self.target_col
        }
    
    @staticmethod
    def get_available_methods():
        """Return list of available aggregation methods."""
        return ['huber', 'trimmed', 'population_weighted', 'mean', 'median']
    
    @staticmethod
    def get_method_description(method):
        """Return description of an aggregation method."""
        descriptions = {
            'huber': 'Huber M-estimator - robust to outliers, combines efficiency of mean with robustness of median',
            'trimmed': 'Trimmed Mean - removes extreme values (20% from each end) before averaging',
            'population_weighted': 'Population-Weighted Mean - weights each city by its population proportion',
            'mean': 'Arithmetic Mean - simple average (baseline)',
            'median': 'Median - middle value, robust to outliers'
        }
        return descriptions.get(method, 'Unknown method')
