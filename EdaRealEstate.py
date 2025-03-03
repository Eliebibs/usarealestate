import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime

# Modify the initial data loading section
print("Loading dataset and creating 20% sample...")
df = pd.read_csv('USARealEstate/realtor-data.zip.csv')

# Create single 20% sample
sample_size = 0.20
current_sample = df.sample(frac=sample_size, random_state=42)
print(f"Working with {len(current_sample):,} rows")

# Split into train/test
train_df, test_df = train_test_split(current_sample, test_size=0.2, random_state=42)

# Get all possible states
all_states = df['state'].unique()

print("\nProcessing training data...")
total_steps = 5
current_step = 0

# 2. Handle missing values in critical columns and remove specified rows
def clean_critical_columns(df):
    # Remove rows with missing prices
    df = df.dropna(subset=['price'])
    
    # Remove rows with missing state (only 8 rows)
    df = df.dropna(subset=['state'])
    
    # Remove rows with missing zip code (only 299 rows)
    df = df.dropna(subset=['zip_code'])
    
    return df

# 3. Handle price outliers (do this early as it affects other calculations)
def handle_price_outliers(df):
    price_low = df['price'].quantile(0.01)  # 1st percentile
    price_high = df['price'].quantile(0.99)  # 99th percentile
    df = df[(df['price'] >= 10000) & (df['price'] <= 3890000)]
    return df

# 4. Handle missing values and outliers in numerical columns
def process_numerical_columns(df):
    # Bed - cap at 7 and fill missing with median
    df['bed'] = df['bed'].clip(upper=7)
    df['bed'] = df['bed'].fillna(df['bed'].median())
    
    # Bath - cap at 6 and fill missing based on bed correlation
    df['bath'] = df['bath'].clip(upper=6)
    median_bath_by_bed = df.groupby('bed')['bath'].median()
    df['bath'] = df.apply(lambda x: median_bath_by_bed[x['bed']] if pd.isna(x['bath']) else x['bath'], axis=1)
    
    # Acre lot - cap at 98 acres and fill missing with zip code median
    df['acre_lot'] = df['acre_lot'].clip(upper=98)
    median_acre_by_zip = df.groupby('zip_code')['acre_lot'].median()
    df['acre_lot'] = df.apply(lambda x: median_acre_by_zip.get(x['zip_code'], df['acre_lot'].median()) 
                             if pd.isna(x['acre_lot']) else x['acre_lot'], axis=1)
    
    # House size - cap at 6439 sqft and fill missing
    df['house_size'] = df['house_size'].clip(upper=6439)
    # Complex filling based on zip_code, bed, and bath
    median_size_by_groups = df.groupby(['zip_code', 'bed', 'bath'])['house_size'].median()
    df['house_size'] = df.apply(lambda x: median_size_by_groups.get((x['zip_code'], x['bed'], x['bath']), 
                               df['house_size'].median()) if pd.isna(x['house_size']) else x['house_size'], axis=1)
    
    return df

# 5. Handle categorical columns
def process_categorical_columns(df, all_states):
    # Brokered by - replace missing with "missing" placeholder
    df['brokered_by'] = df['brokered_by'].fillna('missing')
    
    # City - replace missing with "missing" and create engineered features
    df['city'] = df['city'].fillna('missing')
    city_freq = df['city'].value_counts(normalize=True)
    city_price_mean = df.groupby('city')['price'].mean()
    df['city_freq'] = df['city'].map(city_freq)
    df['city_price_mean'] = df['city'].map(city_price_mean)
    
    # Zip code - create engineered features
    zip_freq = df['zip_code'].value_counts(normalize=True)
    zip_price_mean = df.groupby('zip_code')['price'].mean()
    df['zip_freq'] = df['zip_code'].map(zip_freq)
    df['zip_price_mean'] = df['zip_code'].map(zip_price_mean)
    
    # State - one-hot encoding with explicit 0/1, ensuring all states are included
    state_dummies = pd.get_dummies(df['state'], prefix='state', dtype=int)
    # Add missing state columns with zeros
    for state in all_states:
        state_col = f'state_{state}'
        if state_col not in state_dummies.columns:
            state_dummies[state_col] = 0
    
    df = pd.concat([df, state_dummies], axis=1)
    
    # Status - one-hot encoding with explicit 0/1
    status_dummies = pd.get_dummies(df['status'], prefix='status', dtype=int)
    df = pd.concat([df, status_dummies], axis=1)
    
    # Previous sold date
    df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
    df['years_since_sold'] = (pd.Timestamp.now() - df['prev_sold_date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df['years_since_sold'] = df['years_since_sold'].fillna(-1)  # -1 for missing values
    
    return df

# 6. Apply transformations
def apply_transformations(df):
    # Log transformation for price
    df['price_log'] = np.log1p(df['price'])
    
    # Log transformation for acre_lot
    df['acre_lot_log'] = np.log1p(df['acre_lot'])
    
    # Log transformation for house_size
    df['house_size_log'] = np.log1p(df['house_size'])
    
    return df

# 7. Scale numerical features
def scale_features(train_df, test_df):
    scaler = MinMaxScaler()
    columns_to_scale = ['bath', 'city_freq', 'zip_freq', 'city_price_mean', 
                       'zip_price_mean', 'acre_lot_log', 'house_size_log']
    
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
    
    return train_df, test_df

# Apply all transformations with progress updates
current_step += 1
print(f"Step {current_step}/{total_steps}: Cleaning critical columns...")
train_df = clean_critical_columns(train_df)
test_df = clean_critical_columns(test_df)

current_step += 1
print(f"Step {current_step}/{total_steps}: Handling price outliers...")
train_df = handle_price_outliers(train_df)
test_df = handle_price_outliers(test_df)

current_step += 1
print(f"Step {current_step}/{total_steps}: Processing numerical columns...")
train_df = process_numerical_columns(train_df)
test_df = process_numerical_columns(test_df)

current_step += 1
print(f"Step {current_step}/{total_steps}: Processing categorical columns...")
train_df = process_categorical_columns(train_df, all_states)
test_df = process_categorical_columns(test_df, all_states)

current_step += 1
print(f"Step {current_step}/{total_steps}: Applying transformations...")
train_df = apply_transformations(train_df)
test_df = apply_transformations(test_df)

# Scale features using training data fit
print("\nScaling features...")
train_df, test_df = scale_features(train_df, test_df)

# Drop unnecessary columns
print("\nDropping unnecessary columns...")
columns_to_drop = ['street', 'brokered_by', 'prev_sold_date', 'state', 'city', 'zip_code', 'status']
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

# Modify the save operation for single sample
train_df.to_csv('USARealEstate/processed_train_20percent.csv', index=False)
test_df.to_csv('USARealEstate/processed_test_20percent.csv', index=False)

print("\nDataset processed and saved!")

# Function to analyze numerical columns
def analyze_numerical(column):
    return {
        'dtype': str(column.dtype),
        'missing': column.isnull().sum(),
        'unique_count': column.nunique(),
        'mean': column.mean(),
        'median': column.median(),
        'mode': column.mode().iloc[0],
        'min': column.min(),
        'max': column.max(),
        'std': column.std()
    }

# Function to analyze categorical columns
def analyze_categorical(column):
    return {
        'dtype': str(column.dtype),
        'missing': column.isnull().sum(),
        'unique_count': column.nunique()
    }

# Basic statistics for each column
print("\n=== BASIC COLUMN STATISTICS ===")
for column_name in df.columns:
    print(f"\n{'='*50}")
    print(f"Column: {column_name}")
    
    if df[column_name].dtype in ['int64', 'float64']:
        stats = analyze_numerical(df[column_name])
        print(f"Data Type: {stats['dtype']}")
        print(f"Missing Values: {stats['missing']}")
        print(f"Unique Values: {stats['unique_count']}")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Median: {stats['median']:.2f}")
        print(f"Mode: {stats['mode']}")
        print(f"Min: {stats['min']}")
        print(f"Max: {stats['max']}")
        print(f"Standard Deviation: {stats['std']:.2f}")
    else:
        stats = analyze_categorical(df[column_name])
        print(f"Data Type: {stats['dtype']}")
        print(f"Missing Values: {stats['missing']}")
        print(f"Unique Values: {stats['unique_count']}")

print("\n=== DATAFRAME INFO ===")
print(df.info())

# Additional statistical analysis (fast)
print("\n=== DETAILED STATISTICS ===")
columns_to_analyze = ['price', 'bed', 'bath', 'acre_lot', 'house_size']

# 1. Percentile analysis
print("\nPercentile Analysis:")
percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
for column in columns_to_analyze:
    print(f"\n{column.upper()} Percentiles:")
    for p in percentiles:
        value = df[column].quantile(p)
        print(f"{p*100}th percentile: {value:,.2f}")

# 2. Skewness and Kurtosis
print("\nSkewness and Kurtosis:")
for column in columns_to_analyze:
    skew = df[column].skew()
    kurt = df[column].kurtosis()
    print(f"\n{column}:")
    print(f"Skewness: {skew:.2f}")
    print(f"Kurtosis: {kurt:.2f}")

# 3. Value counts for discrete variables
print("\nValue Counts for Discrete Variables:")
for column in ['bed', 'bath']:
    print(f"\n{column.upper()} Distribution:")
    print(df[column].value_counts().sort_index().head(10))

# 4. Range analysis
print("\nRange Analysis:")
for column in columns_to_analyze:
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    print(f"\n{column}:")
    print(f"IQR: {iqr:,.2f}")
    print(f"Range: {df[column].max() - df[column].min():,.2f}")

# Calculate correlations with price using a random sample
sample_size = int(len(df) * 0.03)  # 3% of the data
df_sample = df.sample(n=sample_size, random_state=42)  # Set random_state for reproducibility

# Get numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

print(f"\nProcessing {sample_size} rows...")
print("Calculating correlations...")

# Initialize counter
processed_rows = 0
update_interval = 1000

# Calculate correlations with progress updates
for i in range(0, len(df_sample), update_interval):
    processed_rows += min(update_interval, len(df_sample) - i)
    if i % update_interval == 0:
        print(f"Processed {processed_rows:,} rows out of {sample_size:,} ({(processed_rows/sample_size)*100:.1f}%)")

# Calculate correlations with price
correlations = df_sample[numerical_columns].corr()['price'].sort_values(ascending=False)

print("\n=== CORRELATIONS WITH PRICE (3% sample) ===")
print(correlations)

# Comment out or remove the visualization sections if not needed
"""
# === VISUALIZATIONS === #
# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme()
sns.set_palette("husl")

# Distribution plots
fig, axes = plt.subplots(5, 1, figsize=(12, 20))
fig.suptitle('Distribution of Key Real Estate Features', fontsize=16, y=1.02)

# List of columns to plot
columns_to_plot = ['price', 'bed', 'bath', 'acre_lot', 'house_size']

# Create distribution plots
for i, column in enumerate(columns_to_plot):
    # Histogram with KDE
    sns.histplot(data=df, x=column, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    
    # Add median and mean lines
    median = df[column].median()
    mean = df[column].mean()
    axes[i].axvline(median, color='red', linestyle='--', label=f'Median: {median:.2f}')
    axes[i].axvline(mean, color='green', linestyle='--', label=f'Mean: {mean:.2f}')
    axes[i].legend()

plt.tight_layout()
plt.show()

# Also create box plots to show outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[columns_to_plot])
plt.xticks(rotation=45)
plt.title('Box Plots of Key Features')
plt.show()

# Add log-transformed plots for skewed distributions
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Log-Transformed Distributions', fontsize=16, y=1.02)

# Price distribution
sns.histplot(data=df, x='price', kde=True, ax=axes[0])
axes[0].set_title('Price Distribution')

# Log-transformed price distribution
sns.histplot(data=df, x='price', kde=True, ax=axes[1], log_scale=True)
axes[1].set_title('Log-Transformed Price Distribution')

plt.tight_layout()
plt.show()

# 5. Violin plots (combines box plot with KDE)
plt.figure(figsize=(15, 6))
for i, column in enumerate(['bed', 'bath']):
    plt.subplot(1, 2, i+1)
    sns.violinplot(data=df, y=column)
    plt.title(f'Violin Plot of {column}')
plt.tight_layout()
plt.show()

# 6. Scatter plots to see relationships
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(df['bed'], df['price'], alpha=0.5)
plt.xlabel('Bedrooms')
plt.ylabel('Price')

plt.subplot(1, 3, 2)
plt.scatter(df['bath'], df['price'], alpha=0.5)
plt.xlabel('Bathrooms')
plt.ylabel('Price')

plt.subplot(1, 3, 3)
plt.scatter(df['house_size'], df['price'], alpha=0.5)
plt.xlabel('House Size')
plt.ylabel('Price')

plt.tight_layout()
plt.show()
# 7. Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[['price', 'bed', 'bath', 'acre_lot', 'house_size']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
"""
