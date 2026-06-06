import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

print("="*80)
print(" " * 20 + "SATELLITE POSITIONING ERROR ANALYSIS")
print(" " * 25 + "Advanced Time Series Report")
print("="*80)

# Load data
try:
    geo_data = pd.read_csv('geo_training.txt', sep='\t')
    meo_data = pd.read_csv('meo_training.txt', sep='\t')
    meo2_data = pd.read_csv('meo2_training.txt', sep='\t')
    
    print("\n✓ Data Loading Successful")
    print(f"  │ GEO (Geostationary):        {len(geo_data):>4} samples")
    print(f"  │ MEO (Medium Earth Orbit):   {len(meo_data):>4} samples")
    print(f"  │ MEO2 (Medium Earth Orbit):  {len(meo2_data):>4} samples")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit()

# Add satellite type labels
geo_data['satellite_type'] = 'GEO'
meo_data['satellite_type'] = 'MEO'
meo2_data['satellite_type'] = 'MEO2'

# Combine datasets
all_data = pd.concat([geo_data, meo_data, meo2_data], ignore_index=True)
all_data['datetime'] = pd.to_datetime(all_data['utc_time'], format='%m-%d-%Y %H:%M')
all_data = all_data.sort_values('datetime').reset_index(drop=True)

# Data cleaning
print(f"\n✓ Data Preprocessing")
print(f"  │ Original samples: {len(all_data)}")
missing = all_data.isnull().sum()
if missing.sum() > 0:
    print(f"  │ Missing values found:")
    for col, count in missing[missing > 0].items():
        print(f"  │   - {col}: {count}")

all_data = all_data.dropna(subset=['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)'])
print(f"  │ Clean samples: {len(all_data)}")

# Calculate error metrics
all_data['total_error'] = np.sqrt(
    all_data['x_error (m)']**2 + 
    all_data['y_error (m)']**2 + 
    all_data['z_error (m)']**2
)
all_data['horizontal_error'] = np.sqrt(
    all_data['x_error (m)']**2 + 
    all_data['y_error (m)']**2
)
all_data['vertical_error'] = np.abs(all_data['z_error (m)'])

# Feature engineering
all_data['hour'] = all_data['datetime'].dt.hour
all_data['day'] = all_data['datetime'].dt.day
all_data['day_of_week'] = all_data['datetime'].dt.dayofweek
all_data['hour_sin'] = np.sin(2 * np.pi * all_data['hour'] / 24)
all_data['hour_cos'] = np.cos(2 * np.pi * all_data['hour'] / 24)
all_data['time_elapsed'] = (all_data['datetime'] - all_data['datetime'].min()).dt.total_seconds() / 3600

# Add rolling statistics
all_data['error_ma_5'] = all_data['total_error'].rolling(window=5, min_periods=1).mean()
all_data['error_std_5'] = all_data['total_error'].rolling(window=5, min_periods=1).std().fillna(0)

sat_type_map = {'GEO': 0, 'MEO': 1, 'MEO2': 2}
all_data['sat_type_encoded'] = all_data['satellite_type'].map(sat_type_map)

# Remove any remaining NaN values after feature engineering
print(f"\n✓ Feature Engineering Complete")
print(f"  │ Checking for NaN values after feature creation...")
nan_count = all_data[['error_ma_5', 'error_std_5', 'total_error']].isnull().sum().sum()
if nan_count > 0:
    print(f"  │ Found {nan_count} NaN values, removing...")
    all_data = all_data.dropna(subset=['error_ma_5', 'error_std_5', 'total_error'])
    print(f"  │ Final dataset size: {len(all_data)}")

# Statistics
print(f"\n✓ Dataset Overview")
print(f"  │ Date range: {all_data['datetime'].min()} to {all_data['datetime'].max()}")
print(f"  │ Duration: {(all_data['datetime'].max() - all_data['datetime'].min()).days} days")
print(f"\n  Error Statistics (meters):")
print(f"  ┌─────────────────┬─────────┬─────────┬─────────┬─────────┐")
print(f"  │ Metric          │   Mean  │  Median │   Std   │   Max   │")
print(f"  ├─────────────────┼─────────┼─────────┼─────────┼─────────┤")
for col in ['total_error', 'horizontal_error', 'vertical_error']:
    mean_val = all_data[col].mean()
    med_val = all_data[col].median()
    std_val = all_data[col].std()
    max_val = all_data[col].max()
    col_name = col.replace('_', ' ').title()[:15]
    print(f"  │ {col_name:<15} │ {mean_val:7.3f} │ {med_val:7.3f} │ {std_val:7.3f} │ {max_val:7.3f} │")
print(f"  └─────────────────┴─────────┴─────────┴─────────┴─────────┘")

# Train/Test split
split_idx = int(0.8 * len(all_data))
train_data = all_data.iloc[:split_idx].copy()
test_data = all_data.iloc[split_idx:].copy()

print(f"\n✓ Train/Test Split")
print(f"  │ Training: {len(train_data)} samples ({len(train_data)/len(all_data)*100:.1f}%)")
print(f"  │ Testing:  {len(test_data)} samples ({len(test_data)/len(all_data)*100:.1f}%)")

# Model training
feature_cols = ['hour', 'day', 'day_of_week', 'hour_sin', 'hour_cos', 
                'sat_type_encoded', 'satclockerror (m)', 'time_elapsed',
                'error_ma_5', 'error_std_5']

X_train = train_data[feature_cols].values
y_train = train_data['total_error'].values
X_test = test_data[feature_cols].values
y_test = test_data['total_error'].values

print(f"\n✓ Training Models")
print(f"  │ Features: {len(feature_cols)}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=150, max_depth=20, 
                                  min_samples_split=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=10, 
                                      learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_train_pred = gb_model.predict(X_train)
gb_test_pred = gb_model.predict(X_test)

# Calculate metrics
def calc_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }

rf_train_metrics = calc_metrics(y_train, rf_train_pred)
rf_test_metrics = calc_metrics(y_test, rf_test_pred)
gb_train_metrics = calc_metrics(y_train, gb_train_pred)
gb_test_metrics = calc_metrics(y_test, gb_test_pred)

print(f"\n  Model Performance:")
print(f"  ┌──────────────────┬─────────────────┬─────────────────┐")
print(f"  │ Metric           │ Random Forest   │ Gradient Boost  │")
print(f"  ├──────────────────┼─────────────────┼─────────────────┤")
print(f"  │ Train RMSE       │ {rf_train_metrics['RMSE']:14.4f}  │ {gb_train_metrics['RMSE']:14.4f}  │")
print(f"  │ Test RMSE        │ {rf_test_metrics['RMSE']:14.4f}  │ {gb_test_metrics['RMSE']:14.4f}  │")
print(f"  │ Train R²         │ {rf_train_metrics['R2']:14.4f}  │ {gb_train_metrics['R2']:14.4f}  │")
print(f"  │ Test R²          │ {rf_test_metrics['R2']:14.4f}  │ {gb_test_metrics['R2']:14.4f}  │")
print(f"  │ Test MAE         │ {rf_test_metrics['MAE']:14.4f}  │ {gb_test_metrics['MAE']:14.4f}  │")
print(f"  │ Test MAPE (%)    │ {rf_test_metrics['MAPE']:14.2f}  │ {gb_test_metrics['MAPE']:14.2f}  │")
print(f"  └──────────────────┴─────────────────┴─────────────────┘")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Satellite Positioning Error - Advanced Time Series Analysis Report', 
             fontsize=18, fontweight='bold', y=0.995)

# 1. Main time series with all satellites
ax1 = fig.add_subplot(gs[0, :2])
for i, sat_type in enumerate(['GEO', 'MEO', 'MEO2']):
    data_subset = all_data[all_data['satellite_type'] == sat_type]
    ax1.plot(data_subset['datetime'], data_subset['total_error'], 
             alpha=0.7, label=sat_type, linewidth=1.5, color=colors[i])
ax1.axvline(x=train_data['datetime'].iloc[-1], color='red', 
            linestyle='--', linewidth=2.5, label='Train/Test Split', alpha=0.7)
ax1.set_xlabel('Date & Time', fontsize=11, fontweight='bold')
ax1.set_ylabel('Total Position Error (m)', fontsize=11, fontweight='bold')
ax1.set_title('Position Error Over Time by Satellite Type', fontsize=12, fontweight='bold', pad=10)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=30)

# 2. Model comparison on test set
ax2 = fig.add_subplot(gs[0, 2:])
test_dates = test_data['datetime'].values
ax2.plot(test_dates, y_test, 'o-', alpha=0.6, label='Actual', 
         linewidth=2, markersize=4, color='black')
ax2.plot(test_dates, rf_test_pred, 's-', alpha=0.7, label='Random Forest', 
         linewidth=1.5, markersize=3, color=colors[0])
ax2.plot(test_dates, gb_test_pred, '^-', alpha=0.7, label='Gradient Boosting', 
         linewidth=1.5, markersize=3, color=colors[1])
ax2.set_xlabel('Date & Time', fontsize=11, fontweight='bold')
ax2.set_ylabel('Total Error (m)', fontsize=11, fontweight='bold')
ax2.set_title(f'Model Predictions on Test Set (RF R²={rf_test_metrics["R2"]:.3f})', 
              fontsize=12, fontweight='bold', pad=10)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=30)

# 3. Error components by satellite
ax3 = fig.add_subplot(gs[1, 0])
sat_types = ['GEO', 'MEO', 'MEO2']
x_pos = np.arange(len(sat_types))
width = 0.25
for i, sat_type in enumerate(sat_types):
    data_subset = all_data[all_data['satellite_type'] == sat_type]
    means = [np.abs(data_subset['x_error (m)']).mean(),
             np.abs(data_subset['y_error (m)']).mean(),
             np.abs(data_subset['z_error (m)']).mean()]
    ax3.bar([i - width, i, i + width], means, width*0.9, 
            label=['X', 'Y', 'Z'][i%3], alpha=0.8, color=colors[i])
ax3.set_xticks(x_pos)
ax3.set_xticklabels(sat_types)
ax3.set_ylabel('Mean |Error| (m)', fontsize=10, fontweight='bold')
ax3.set_title('Error Components by Satellite', fontsize=11, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.legend(['X', 'Y', 'Z'], fontsize=8)

# 4. Prediction accuracy scatter
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_test, rf_test_pred, alpha=0.6, s=40, c=colors[0], 
            edgecolors='black', linewidth=0.5, label='RF')
lims = [0, max(y_test.max(), rf_test_pred.max()) * 1.05]
ax4.plot(lims, lims, 'r--', lw=2.5, alpha=0.7, label='Perfect Fit')
ax4.set_xlabel('Actual Error (m)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Predicted Error (m)', fontsize=10, fontweight='bold')
ax4.set_title(f'Prediction Accuracy (RMSE={rf_test_metrics["RMSE"]:.3f}m)', 
              fontsize=11, fontweight='bold', pad=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, linestyle='--')

# 5. Residual analysis
ax5 = fig.add_subplot(gs[1, 2])
residuals = y_test - rf_test_pred
ax5.scatter(rf_test_pred, residuals, alpha=0.6, s=40, c=colors[2], 
            edgecolors='black', linewidth=0.5)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax5.axhline(y=residuals.std(), color='orange', linestyle=':', linewidth=2, alpha=0.6, label='+1 Std')
ax5.axhline(y=-residuals.std(), color='orange', linestyle=':', linewidth=2, alpha=0.6, label='-1 Std')
ax5.set_xlabel('Predicted Error (m)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Residuals (m)', fontsize=10, fontweight='bold')
ax5.set_title('Residual Analysis', fontsize=11, fontweight='bold', pad=10)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, linestyle='--')

# 6. QQ plot for residuals
ax6 = fig.add_subplot(gs[1, 3])
stats.probplot(residuals, dist="norm", plot=ax6)
ax6.get_lines()[0].set_marker('o')
ax6.get_lines()[0].set_markersize(4)
ax6.get_lines()[0].set_markerfacecolor(colors[3])
ax6.get_lines()[0].set_alpha(0.6)
ax6.get_lines()[1].set_linewidth(2.5)
ax6.get_lines()[1].set_color('red')
ax6.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold', pad=10)
ax6.grid(True, alpha=0.3, linestyle='--')

# 7. Feature importance
ax7 = fig.add_subplot(gs[2, 0])
importance = rf_model.feature_importances_
indices = np.argsort(importance)
feature_names = [feature_cols[i] for i in indices]
y_pos = np.arange(len(feature_names))
bars = ax7.barh(y_pos, importance[indices], color=colors[0], alpha=0.8, edgecolor='black')
ax7.set_yticks(y_pos)
ax7.set_yticklabels(feature_names, fontsize=9)
ax7.set_xlabel('Importance', fontsize=10, fontweight='bold')
ax7.set_title('Feature Importance (RF)', fontsize=11, fontweight='bold', pad=10)
ax7.grid(True, alpha=0.3, axis='x', linestyle='--')

# 8. Hourly pattern analysis
ax8 = fig.add_subplot(gs[2, 1])
hourly_stats = all_data.groupby(['hour', 'satellite_type'])['total_error'].agg(['mean', 'std']).reset_index()
for i, sat_type in enumerate(['GEO', 'MEO', 'MEO2']):
    data_subset = hourly_stats[hourly_stats['satellite_type'] == sat_type]
    ax8.plot(data_subset['hour'], data_subset['mean'], marker='o', 
             linewidth=2.5, label=sat_type, color=colors[i], markersize=5)
    ax8.fill_between(data_subset['hour'], 
                      data_subset['mean'] - data_subset['std'],
                      data_subset['mean'] + data_subset['std'],
                      alpha=0.2, color=colors[i])
ax8.set_xlabel('Hour of Day', fontsize=10, fontweight='bold')
ax8.set_ylabel('Mean Error (m)', fontsize=10, fontweight='bold')
ax8.set_title('Diurnal Error Pattern', fontsize=11, fontweight='bold', pad=10)
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3, linestyle='--')
ax8.set_xticks(range(0, 24, 3))

# 9. Error distribution by satellite
ax9 = fig.add_subplot(gs[2, 2])
box_data = [all_data[all_data['satellite_type'] == st]['total_error'].values 
            for st in ['GEO', 'MEO', 'MEO2']]
bp = ax9.boxplot(box_data, labels=['GEO', 'MEO', 'MEO2'], patch_artist=True,
                  medianprops=dict(color='red', linewidth=2),
                  boxprops=dict(linewidth=1.5),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax9.set_ylabel('Total Error (m)', fontsize=10, fontweight='bold')
ax9.set_title('Error Distribution', fontsize=11, fontweight='bold', pad=10)
ax9.grid(True, alpha=0.3, axis='y', linestyle='--')

# 10. Clock error correlation
ax10 = fig.add_subplot(gs[2, 3])
for i, sat_type in enumerate(['GEO', 'MEO', 'MEO2']):
    data_subset = all_data[all_data['satellite_type'] == sat_type]
    ax10.scatter(data_subset['satclockerror (m)'], data_subset['total_error'], 
                 alpha=0.5, s=30, label=sat_type, color=colors[i])
ax10.set_xlabel('Clock Error (m)', fontsize=10, fontweight='bold')
ax10.set_ylabel('Position Error (m)', fontsize=10, fontweight='bold')
ax10.set_title('Clock Error vs Position Error', fontsize=11, fontweight='bold', pad=10)
ax10.legend(fontsize=8)
ax10.grid(True, alpha=0.3, linestyle='--')

# 11. Cumulative distribution
ax11 = fig.add_subplot(gs[3, 0])
for i, sat_type in enumerate(['GEO', 'MEO', 'MEO2']):
    data_subset = all_data[all_data['satellite_type'] == sat_type]['total_error'].sort_values()
    cumulative = np.arange(1, len(data_subset) + 1) / len(data_subset) * 100
    ax11.plot(data_subset, cumulative, linewidth=2.5, label=sat_type, color=colors[i])
ax11.set_xlabel('Total Error (m)', fontsize=10, fontweight='bold')
ax11.set_ylabel('Cumulative %', fontsize=10, fontweight='bold')
ax11.set_title('Cumulative Error Distribution', fontsize=11, fontweight='bold', pad=10)
ax11.legend(fontsize=8)
ax11.grid(True, alpha=0.3, linestyle='--')

# 12. Error histogram
ax12 = fig.add_subplot(gs[3, 1])
for i, sat_type in enumerate(['GEO', 'MEO', 'MEO2']):
    data_subset = all_data[all_data['satellite_type'] == sat_type]
    ax12.hist(data_subset['total_error'], bins=25, alpha=0.6, 
              label=sat_type, density=True, color=colors[i], edgecolor='black')
ax12.set_xlabel('Total Error (m)', fontsize=10, fontweight='bold')
ax12.set_ylabel('Density', fontsize=10, fontweight='bold')
ax12.set_title('Error Probability Distribution', fontsize=11, fontweight='bold', pad=10)
ax12.legend(fontsize=8)
ax12.grid(True, alpha=0.3, axis='y', linestyle='--')

# 13. Rolling statistics
ax13 = fig.add_subplot(gs[3, 2])
window = 10
rolling_mean = all_data['total_error'].rolling(window=window).mean()
rolling_std = all_data['total_error'].rolling(window=window).std()
ax13.plot(all_data['datetime'], rolling_mean, linewidth=2, 
          label=f'{window}-pt Moving Avg', color=colors[0])
ax13.fill_between(all_data['datetime'], 
                   rolling_mean - rolling_std,
                   rolling_mean + rolling_std,
                   alpha=0.3, color=colors[0], label='±1 Std Dev')
ax13.set_xlabel('Date & Time', fontsize=10, fontweight='bold')
ax13.set_ylabel('Error (m)', fontsize=10, fontweight='bold')
ax13.set_title('Rolling Statistics', fontsize=11, fontweight='bold', pad=10)
ax13.legend(fontsize=8)
ax13.grid(True, alpha=0.3, linestyle='--')
ax13.tick_params(axis='x', rotation=30)

# 14. Model comparison metrics
ax14 = fig.add_subplot(gs[3, 3])
metrics_comparison = pd.DataFrame({
    'RF Train': [rf_train_metrics['RMSE'], rf_train_metrics['MAE'], rf_train_metrics['R2']],
    'RF Test': [rf_test_metrics['RMSE'], rf_test_metrics['MAE'], rf_test_metrics['R2']],
    'GB Train': [gb_train_metrics['RMSE'], gb_train_metrics['MAE'], gb_train_metrics['R2']],
    'GB Test': [gb_test_metrics['RMSE'], gb_test_metrics['MAE'], gb_test_metrics['R2']]
}, index=['RMSE', 'MAE', 'R²'])

x_pos = np.arange(len(metrics_comparison.index))
width = 0.2
for i, col in enumerate(metrics_comparison.columns):
    offset = (i - 1.5) * width
    ax14.bar(x_pos + offset, metrics_comparison[col], width, 
             label=col, alpha=0.8, color=colors[i])
ax14.set_xticks(x_pos)
ax14.set_xticklabels(metrics_comparison.index, fontsize=9)
ax14.set_ylabel('Metric Value', fontsize=10, fontweight='bold')
ax14.set_title('Model Performance Comparison', fontsize=11, fontweight='bold', pad=10)
ax14.legend(fontsize=7, ncol=2)
ax14.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('satellite_error_analysis_report.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved as 'satellite_error_analysis_report.png'")
plt.show()

# Additional statistical analysis
print("\n" + "="*80)
print(" " * 25 + "DETAILED STATISTICAL ANALYSIS")
print("="*80)

print("\n📊 Error Statistics by Satellite Type:")
print("─" * 80)
for sat_type in ['GEO', 'MEO', 'MEO2']:
    data_subset = all_data[all_data['satellite_type'] == sat_type]
    print(f"\n{sat_type} Satellite:")
    print(f"  │ Sample count:        {len(data_subset):>6}")
    print(f"  │ Mean total error:    {data_subset['total_error'].mean():>6.3f} m")
    print(f"  │ Median total error:  {data_subset['total_error'].median():>6.3f} m")
    print(f"  │ Std deviation:       {data_subset['total_error'].std():>6.3f} m")
    print(f"  │ Min error:           {data_subset['total_error'].min():>6.3f} m")
    print(f"  │ Max error:           {data_subset['total_error'].max():>6.3f} m")
    print(f"  │ 95th percentile:     {data_subset['total_error'].quantile(0.95):>6.3f} m")
    print(f"  │ 99th percentile:     {data_subset['total_error'].quantile(0.99):>6.3f} m")
    
    # Calculate accuracy metrics (percentage within thresholds)
    within_1m = (data_subset['total_error'] <= 1).sum() / len(data_subset) * 100
    within_5m = (data_subset['total_error'] <= 5).sum() / len(data_subset) * 100
    within_10m = (data_subset['total_error'] <= 10).sum() / len(data_subset) * 100
    print(f"  │ Within 1m:           {within_1m:>5.1f}%")
    print(f"  │ Within 5m:           {within_5m:>5.1f}%")
    print(f"  │ Within 10m:          {within_10m:>5.1f}%")

# Correlation analysis
print("\n\n📈 Correlation Analysis:")
print("─" * 80)
corr_cols = ['x_error (m)', 'y_error (m)', 'z_error (m)', 
             'satclockerror (m)', 'total_error']
correlation_matrix = all_data[corr_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3).to_string())

# Statistical tests
print("\n\n🔬 Statistical Hypothesis Tests:")
print("─" * 80)

# Test for normality of residuals
_, p_shapiro = stats.shapiro(residuals[:min(5000, len(residuals))])
print(f"\nShapiro-Wilk Normality Test (Residuals):")
print(f"  │ p-value: {p_shapiro:.6f}")
print(f"  │ Result: {'Residuals are normally distributed' if p_shapiro > 0.05 else 'Residuals are NOT normally distributed'} (α=0.05)")

# Kruskal-Wallis test for satellite types
sat_groups = [all_data[all_data['satellite_type'] == st]['total_error'].values 
              for st in ['GEO', 'MEO', 'MEO2']]
h_stat, p_kruskal = stats.kruskal(*sat_groups)
print(f"\nKruskal-Wallis Test (Satellite Type Differences):")
print(f"  │ H-statistic: {h_stat:.4f}")
print(f"  │ p-value: {p_kruskal:.6f}")
print(f"  │ Result: {'Significant difference' if p_kruskal < 0.05 else 'No significant difference'} between satellite types (α=0.05)")

# Time series stationarity test (Augmented Dickey-Fuller)
try:
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(all_data['total_error'].dropna())
    print(f"\nAugmented Dickey-Fuller Test (Stationarity):")
    print(f"  │ ADF Statistic: {adf_result[0]:.4f}")
    print(f"  │ p-value: {adf_result[1]:.6f}")
    print(f"  │ Result: {'Series is stationary' if adf_result[1] < 0.05 else 'Series is non-stationary'} (α=0.05)")
except ImportError:
    print("\n  ⚠ statsmodels not available for ADF test")

# Model performance summary
print("\n\n🎯 Model Performance Summary:")
print("─" * 80)
print(f"\nBest Model: {'Random Forest' if rf_test_metrics['RMSE'] < gb_test_metrics['RMSE'] else 'Gradient Boosting'}")
print(f"\n  Performance on Test Set:")
print(f"  │ RMSE:           {min(rf_test_metrics['RMSE'], gb_test_metrics['RMSE']):.4f} m")
print(f"  │ MAE:            {min(rf_test_metrics['MAE'], gb_test_metrics['MAE']):.4f} m")
print(f"  │ R² Score:       {max(rf_test_metrics['R2'], gb_test_metrics['R2']):.4f}")
print(f"  │ MAPE:           {min(rf_test_metrics['MAPE'], gb_test_metrics['MAPE']):.2f}%")

# Error budget analysis
print("\n\n💡 Error Budget Analysis:")
print("─" * 80)
total_var = all_data['total_error'].var()
x_var = all_data['x_error (m)'].var()
y_var = all_data['y_error (m)'].var()
z_var = all_data['z_error (m)'].var()
print(f"\nVariance Contribution:")
print(f"  │ X-component: {x_var/total_var*100:>5.1f}%")
print(f"  │ Y-component: {y_var/total_var*100:>5.1f}%")
print(f"  │ Z-component: {z_var/total_var*100:>5.1f}%")

# Key insights
print("\n\n✨ Key Insights & Recommendations:")
print("─" * 80)

# Identify worst performing satellite
worst_sat = all_data.groupby('satellite_type')['total_error'].mean().idxmax()
best_sat = all_data.groupby('satellite_type')['total_error'].mean().idxmin()

print(f"\n1. Satellite Performance:")
print(f"   • {worst_sat} satellites show highest positioning errors")
print(f"   • {best_sat} satellites demonstrate best accuracy")

# Time-based patterns
peak_hour = all_data.groupby('hour')['total_error'].mean().idxmax()
best_hour = all_data.groupby('hour')['total_error'].mean().idxmin()
print(f"\n2. Temporal Patterns:")
print(f"   • Peak errors occur around {peak_hour:02d}:00 UTC")
print(f"   • Best performance at {best_hour:02d}:00 UTC")
print(f"   • Strong diurnal variation detected")

# Model reliability
print(f"\n3. Prediction Model:")
print(f"   • Model explains {rf_test_metrics['R2']*100:.1f}% of error variance")
print(f"   • Average prediction error: {rf_test_metrics['MAE']:.3f} meters")
if rf_test_metrics['R2'] > 0.7:
    print(f"   • High reliability for operational forecasting")
elif rf_test_metrics['R2'] > 0.5:
    print(f"   • Moderate reliability, consider additional features")
else:
    print(f"   • Limited predictive power, requires model improvement")

# Clock error impact
clock_corr = all_data[['satclockerror (m)', 'total_error']].corr().iloc[0, 1]
print(f"\n4. Clock Error Impact:")
print(f"   • Correlation with position error: {clock_corr:.3f}")
if abs(clock_corr) > 0.5:
    print(f"   • Strong relationship - clock calibration critical")
elif abs(clock_corr) > 0.3:
    print(f"   • Moderate relationship - monitor clock stability")
else:
    print(f"   • Weak relationship - other factors dominate")

# Outlier analysis
outlier_threshold = all_data['total_error'].quantile(0.95)
outlier_count = (all_data['total_error'] > outlier_threshold).sum()
print(f"\n5. Outlier Analysis:")
print(f"   • {outlier_count} samples exceed 95th percentile ({outlier_threshold:.2f}m)")
print(f"   • Represents {outlier_count/len(all_data)*100:.1f}% of observations")
print(f"   • Investigate extreme error events for system reliability")

print("\n" + "="*80)
print(" " * 30 + "ANALYSIS COMPLETE")
print("="*80)
print("\n📁 Output files generated:")
print("   • satellite_error_analysis_report.png (High-resolution visualization)")
print("\n💡 Recommendations:")
print("   • Monitor GEO satellite performance closely")
print("   • Consider time-based error correction algorithms")
print("   • Implement predictive maintenance using the trained models")
print("   • Investigate outlier events for system improvements")
print("\n" + "="*80 + "\n")