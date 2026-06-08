import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
df = pd.read_csv('WorldCupMatches.csv')

# 2. DATA CLEANING: Duplicates & Missing Values
df.drop_duplicates(inplace=True)
df.dropna(how='all', inplace=True) # Drop completely empty rows

# Impute missing attendance with median
if 'Attendance' in df.columns:
    df['Attendance'] = df['Attendance'].fillna(df['Attendance'].median())

# 3. DATA PROCESSING
df['Year'] = df['Year'].astype(int)
df['Total Goals'] = df['Home Team Goals'] + df['Away Team Goals']

# Outlier Detection (IQR Method on Attendance)
Q1 = df['Attendance'].quantile(0.25)
Q3 = df['Attendance'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
outliers_count = df[df['Attendance'] > upper_bound].shape[0]
print(f"Attendance outliers identified: {outliers_count}")

# 4. VISUALIZATION DASHBOARD
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('FIFA World Cup Matches: Data Insights Dashboard', fontsize=20, fontweight='bold')

# Plot A: Total Goals per Year
goals_per_year = df.groupby('Year')['Total Goals'].sum().reset_index()
sns.barplot(data=goals_per_year, x='Year', y='Total Goals', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Total Goals Scored per World Cup Year', fontsize=14)
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
axes[0, 0].set_ylabel('Total Goals')

# Plot B: Average Attendance per Year
attendance_per_year = df.groupby('Year')['Attendance'].mean().reset_index()
sns.lineplot(data=attendance_per_year, x='Year', y='Attendance', marker='o', ax=axes[0, 1], color='coral', linewidth=2.5)
axes[0, 1].set_title('Average Attendance per World Cup Year', fontsize=14)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)

# Plot C: Attendance Boxplot (Outliers)
sns.boxplot(data=df, x='Attendance', ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('Distribution of Match Attendance (Outlier Check)', fontsize=14)

# Plot D: Top 10 Teams by Goals
home_goals = df.groupby('Home Team Name')['Home Team Goals'].sum().reset_index().rename(columns={'Home Team Name': 'Team', 'Home Team Goals': 'Goals'})
away_goals = df.groupby('Away Team Name')['Away Team Goals'].sum().reset_index().rename(columns={'Away Team Name': 'Team', 'Away Team Goals': 'Goals'})
total_team_goals = pd.concat([home_goals, away_goals]).groupby('Team')['Goals'].sum().reset_index().sort_values(by='Goals', ascending=False).head(10)

sns.barplot(data=total_team_goals, x='Goals', y='Team', ax=axes[1, 1], palette='magma')
axes[1, 1].set_title('Top 10 Teams by Total Goals Scored', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()