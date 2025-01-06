# Python Project: FIFA 22 Analysis
<p align="center">
  <img src="https://www.fifaultimateteam.it/en/wp-content/uploads/2021/06/cover-fifa-22.jpg" />
</p>

## Description
This project uses Python to analyze the dataset of FIFA 22 players. The goal of the project is to answer key questions about player performance, market trends, and value insights, ultimately supporting better decision-making for clubs, investors, and researchers in the football industry.

## Packages
1. **Data Processing and Analysis:**  
   - **`pandas`**: Facilitates manipulation of tabular data (DataFrames), data cleaning, and transformation.  
   - **`numpy`**: Enhances performance with numerical operations and array handling.  

2. **Data Visualization:**  
   - **`matplotlib`**: Used for creating static visualizations like line charts, bar charts, and scatter plots.  
   - **`seaborn`**: Offers aesthetically pleasing and easy-to-use statistical charts.  
   - **`plotly.express`**: Enables interactive visualizations for more flexible data exploration.  

3. **Geospatial Data Analysis:**  
   - **`geopandas`**: Supports analyzing geospatial data, such as mapping player distribution by country.  

4. **Machine Learning and Data Exploration:**  
   - **`sklearn.datasets`**: Provides sample datasets for testing and implementing machine learning models.  

5. **Natural Language Processing:**  
   - **`nltk`**: Offers tools for analyzing and processing text, particularly useful for handling descriptive or textual data.

  
## Prepare Data for Exploration
The <a href="https://github.com/Tbangh16/FIFA-Analyze/blob/master/players_22.csv">dataset</a> players_22.csv includes over 100 columns with detailed information on player attributes, positions, clubs, and countries.

<details>
<summary>Click to show code</summary>

```r
# Data Import
df <- read.csv("players_22.csv", encoding = "UTF-8")[-1]
head(df)
```
```r
# Display general information about the DataFrame
print("General information about the DataFrame:") 
print(df.info()) 
# Display descriptive statistics
print("\nDescriptive Statistics:") 
print(df.describe()) 
# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:") 
print(df.head())

```
</details>


## Process Data from Dirty to Clean

<details>
<summary>Click to show code</summary> 
  
```r
# Plot missing values ratio
missing = df.isnull().mean() * 100
missing = missing[missing > 0].sort_values()
plt.figure(figsize=(10, 6))
colors = plt.cm.Reds(np.linspace(0.3, 1, len(missing)))
missing.plot(kind='bar', color=colors)
plt.title('Percentage of Missing Values', fontsize=16, color='darkred')
plt.xlabel('Columns', fontsize=12, color='darkred')
plt.ylabel('Percentage', fontsize=12, color='darkred')
plt.show()

# Plot count of each data type
plt.figure(figsize=(10, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', 'pink', 'yellow']
df.dtypes.value_counts().plot(kind='bar', color=colors)
plt.title('Count of Data Types', fontsize=16, color='darkorange')
plt.xlabel('Data Types', fontsize=12, color='darkorange')
plt.ylabel('Count', fontsize=12, color='darkorange')
plt.show()

# Plot count of unique values in each column
plt.figure(figsize=(14, 10))
colors = plt.cm.Blues(np.linspace(0.3, 1, len(df.nunique())))
df.nunique().sort_values(ascending=False).plot(kind='bar', color=colors)
plt.title('Count of Unique Values per Column', fontsize=16, color='darkgreen')
plt.xlabel('Columns', fontsize=12, color='darkgreen')
plt.ylabel('Unique Values Count', fontsize=12, color='darkgreen')
plt.show()

```

</details>


### Step 1 :Tạo từ điển mapping cho các league và country

<details>
<summary>Click to show code</summary>

```r
league_country_mapping = {
    'Bundesliga': ('Germany', bundesliga),
    'Premier League': ('UK', premierLeague),
    'La Liga': ('Spain', laliga),
    'Serie A': ('Italy', seriea),
    'Süper Lig': ('Turkey', superlig),
    'Ligue 1': ('France', ligue1),
    'Liga Nos': ('Portugal', liganos),
    'Eredivisie': ('Netherlands', eredivisie)
}

# Hàm để tìm league và country dựa trên tên club
def get_league_and_country(club_name):
    for league, (country, clubs) in league_country_mapping.items():
        if club_name in clubs:
            return league, country
    return None, None

# Áp dụng hàm để tạo các cột 'League' và 'Country'
df['League'], df['Country'] = zip(*df['club_name'].apply(get_league_and_country))

# Lọc ra các dòng có giá trị 'League' là None
df = df.dropna(subset=['League'])

# Hiển thị vài dòng đầu tiên của DataFrame
print(df.head())
```

</details>

### Step 2 : Xử lý các cột đơn vị tiền tệ
<details>
<summary>Click to show code</summary>

```r
# Kiểm tra các kiểu dữ liệu của các cột liên quan
print(df[['value_eur', 'wage_eur']].dtypes)

# Chuyển đổi cột 'value_eur' và 'wage_eur' thành chuỗi nếu cần thiết
df['value_eur'] = df['value_eur'].astype(str)
df['wage_eur'] = df['wage_eur'].astype(str)

# Xử lý cột 'Value' và 'Wage'
df['Values'] = df['value_eur'].str.replace('€', '').str.replace('K', '000').str.replace('M', '').astype(float)
df['Wages'] = df['wage_eur'].str.replace('€', '').str.replace('K', '000').astype(float)

# Chuyển đổi giá trị 'Values' từ triệu sang đơn vị euro
df['Values'] = df['Values'].apply(lambda x: x * 1000000 if x < 1000 else x)

# Hiển thị những dòng đầu tiên của DataFrame để kiểm tra
print(df[['value_eur', 'Values', 'wage_eur', 'Wages']].head())
```

</details>

### Step 3 : Phân loại cầu thủ

<details>
<summary>Click to show code</summary>

```r
defence = ["CB", "RB", "LB", "LWB", "RWB", "LCB", "RCB"]
midfielder = ["CM", "CDM","CAM","LM","RM", "LAM", "RAM", "LCM", "RCM", "LDM", "RDM"]

# Phân loại cầu thủ
df['Class'] = df['club_position'].apply(lambda x: 'Goal Keeper' if x == "GK" else
                                                  'Defender' if x in defence else
                                                  'Midfielder' if x in midfielder else
                                                  'Forward')

# Hiển thị những dòng đầu tiên của DataFrame để kiểm tra
print(df[['club_position', 'Class']].head(15))
```

</details>


### Distribution & The Average Age of The Players in each League
<details>
<summary>Click to show code</summary>

```r
# Calculate average age by League
summ = df.groupby('league_name').agg({'age': 'mean'}).reset_index()

# Set the size and style of the plot
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

# Create a color palette for the leagues
leagues = df['league_name'].unique()
palette = sns.color_palette("husl", len(leagues))
colors = dict(zip(leagues, palette))

# Create histograms and draw the average age line for each league
g = sns.FacetGrid(df, col="league_name", col_wrap=4, sharex=False, sharey=False, palette=palette)

# Plot with distinct colors for each league
for ax, league_name, color in zip(g.axes.flatten(), leagues, palette):
    subset = df[df['league_name'] == league_name]
    sns.histplot(subset, x='age', ax=ax, binwidth=1, color=color)
    # Draw the average age line
    mean_age = summ[summ['league_name'] == league_name]['age'].values[0]
    ax.axvline(mean_age, color='red', linewidth=1.5)
    ax.text(mean_age + 0.5, ax.get_ylim()[1] * 0.9, round(mean_age, 2), color='red')
    # Bold the titles of the plots
    ax.set_title(league_name, fontsize=14, weight='bold')

# Set the plot parameters
g.set_axis_labels("Age", "Frequency")
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Distribution & The Average Age of The Players in each League', fontsize=16, weight='bold')
plt.show()

```

</details>

### Average, Oldest, and Youngest Age of Players by Country
<details>
<summary>Click to show code</summary>

```r
# Calculate age statistics
age_stats = df.groupby('Country')['age'].agg(['mean', 'min', 'max']).reset_index()

# Set the size and style of the plot
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

# Plot with different colors for each country
barplot = sns.barplot(x='Country', y='mean', hue='Country', data=age_stats, palette='viridis', legend=False)

# Add lines for the minimum and maximum ages
for index, row in age_stats.iterrows():
    plt.plot([index, index], [row['min'], row['max']], color='black', linewidth=1.5)
    plt.plot(index, row['min'], 'o', color='blue', markersize=10)
    plt.plot(index, row['max'], 'o', color='red', markersize=10)
    # Add the specific average value on each column
    plt.text(index, row['mean'], f'{row["mean"]:.2f}', color='black', ha='center', fontsize=12, weight='bold')

# Add featured players
oldest_youngest_players = []
for index, country in enumerate(age_stats['Country']):
    subset = df[df['Country'] == country]
    oldest_player = subset.loc[subset['age'].idxmax()]
    youngest_player = subset.loc[subset['age'].idxmin()]
    oldest_youngest_players.append((index, oldest_player['age'], oldest_player['short_name'], 'red'))
    oldest_youngest_players.append((index, youngest_player['age'], youngest_player['short_name'], 'blue'))

# Draw annotation boxes for the featured players
for index, age, name, color in oldest_youngest_players:
    player_info = f'{name}\n({age} yrs)'
    plt.text(index, age, player_info, color=color, ha='center', fontsize=10, weight='bold', 
             bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', alpha=0.8))

# Set plot parameters
plt.title('Average, Oldest, and Youngest Age of Players by Country', fontsize=16, weight='bold')
plt.xlabel('Country', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()


```

</details>

### Market Values of the Leagues
<details>
<summary>Click to show code</summary>

```r
# Clean data: Remove invalid characters and convert to float type
df['value_eur'] = df['value_eur'].replace(r'[\$,]', '', regex=True).astype(float)

# Calculate total market value by League
summ = df.groupby('league_name').agg({'value_eur': 'sum'}).reset_index()

# Set the size of the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Plot bar chart with different colors for each league
bar_plot = sns.barplot(
    x='value_eur', 
    y='league_name', 
    hue='league_name', 
    data=summ.sort_values('value_eur', ascending=False), 
    palette='viridis', 
    dodge=False, 
    legend=False
)

# Set the plot parameters
bar_plot.set_xlabel('Market Values (in Billions €)', fontsize=12)
bar_plot.set_ylabel('Leagues', fontsize=12)
bar_plot.set_title('Market Values of the Leagues', fontsize=16, weight='bold')
bar_plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f} Billion €".format(x / 1e9)))

# Apply theme style
plt.xticks(color='darkslategray', fontsize=10)
plt.yticks(color='darkslategray', fontsize=10)
bar_plot.xaxis.grid(True)
bar_plot.yaxis.grid(False)

# Display the plot
plt.tight_layout()
plt.show()



```

</details>

![Player Skills](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%203%20Teams%20with%20Highest%20Value%20in%20Each%20League.png "Top 3 Teams with Highest Value in Each League")


### Top 3 Teams with Highest Value in Each League
<details>
<summary>Click to show code</summary>

```r
# Set the size and style of the plot
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

# Plot the top 3 teams with the highest value in each league
bar_plot = sns.barplot(
    x='value_eur', 
    y='club_name', 
    hue='league_name', 
    data=top_teams.sort_values('value_eur', ascending=False), 
    palette='viridis'
)

# Set the plot parameters
bar_plot.set_xlabel('Market Values (in €)', fontsize=12)
bar_plot.set_ylabel('Teams', fontsize=12)
bar_plot.set_title('Top 3 Teams with Highest Value in Each League', fontsize=16, weight='bold')
bar_plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f} €".format(x)))

plt.legend(title='Leagues', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




```

</details>

![Player Skills](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%203%20Teams%20with%20Highest%20Value%20in%20Each%20League.png "Top 3 Teams with Highest Value in Each League")


### Top 30 Most Expensive Football Players in the World
<details>
<summary>Click to show code</summary>

```r
# Assume df is a DataFrame containing existing data
# Filter the top 30 most valuable players
top_30_players = df.sort_values(by='value_eur', ascending=False).head(30)

# Select the necessary columns
top_30_players = top_30_players[['short_name', 'club_name', 'value_eur']]

# Set the size and style of the plot
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

# Draw the plot for the top 30 most valuable players in the world
bar_plot = sns.barplot(
    x='value_eur', 
    y='short_name', 
    hue='club_name', 
    data=top_30_players, 
    dodge=False, 
    palette='viridis'
)

# Set the parameters for the plot
bar_plot.set_xlabel('Market Values (in €)', fontsize=12)
bar_plot.set_ylabel('Players', fontsize=12)
bar_plot.set_title('Top 30 Most Expensive Football Players in the World', fontsize=16, weight='bold')
bar_plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f} €".format(x)))

plt.legend(title='Clubs', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```

</details>

![Top 30 Most Expensive Football Players in the World](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%2030%20Most%20Expensive%20Football%20Players%20in%20the%20World.png "Top 30 Most Expensive Football Players in the World")

### Top 30 Most Expensive Football Players in the World
<details>
<summary>Click to show code</summary>

```r
# Assume df is a DataFrame containing existing data
# Filter the top 30 most valuable players
top_30_players = df.sort_values(by='value_eur', ascending=False).head(30)

# Select the necessary columns
top_30_players = top_30_players[['short_name', 'club_name', 'value_eur']]

# Set the size and style of the plot
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

# Draw the plot for the top 30 most valuable players in the world
bar_plot = sns.barplot(
    x='value_eur', 
    y='short_name', 
    hue='club_name', 
    data=top_30_players, 
    dodge=False, 
    palette='viridis'
)

# Set the parameters for the plot
bar_plot.set_xlabel('Market Values (in €)', fontsize=12)
bar_plot.set_ylabel('Players', fontsize=12)
bar_plot.set_title('Top 30 Most Expensive Football Players in the World', fontsize=16, weight='bold')
bar_plot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f} €".format(x)))

plt.legend(title='Clubs', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```

</details>

![Top 30 Most Expensive Football Players in the World](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%2030%20Most%20Expensive%20Football%20Players%20in%20the%20World.png "Top 30 Most Expensive Football Players in the World")


### Top 20 players with the highest Skill
<details>
<summary>Click to show code</summary>

```r
# Convert the relevant columns
df['Skill'] = df['overall']  # Assuming 'overall' is equivalent to 'Skill'
df['Exp'] = df['potential']  # Assuming 'potential' is equivalent to 'Exp'
df['Name'] = df['short_name']  # Assuming 'short_name' is equivalent to 'Name'

# Get the top 20 players with the highest Skill
top_players = df.nlargest(20, 'Skill')

# Set the size of the plot
plt.figure(figsize=(15, 8))

# Plot the bar chart with hue set to Name
g = sns.barplot(x='Skill', y='Name', hue='Name', data=top_players, palette='viridis', dodge=False, legend=False)
g.set_xlabel('Skill')
g.set_ylabel('Name')

# Invert the Y axis to display from top to bottom
plt.gca().invert_yaxis()

# Show the plot
plt.show()
```

</details>

![Top 20 players with the highest Skill](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%2020%20players%20with%20the%20highest%20Skill.png "Top 20 players with the highest Skill")


### Top 30 Most Expensive Football Players in the World
<details>
<summary>Click to show code</summary>

```r
# Filter data for each player
messi = players[players['Name'].str.contains('Messi')]
lewandowski = players[players['Name'].str.contains('Lewandowski')]
ronaldo = players[players['Name'].str.contains('Ronaldo')]

# Pivot the data
messi_skills = messi.pivot(index='Skill', columns='Name', values='Exp').reset_index()
lewandowski_skills = lewandowski.pivot(index='Skill', columns='Name', values='Exp').reset_index()
ronaldo_skills = ronaldo.pivot(index='Skill', columns='Name', values='Exp').reset_index()

# Create radar chart function
def create_radar_chart(df, title, ax):
    categories = list(df['Skill'])
    labels = list(df.columns[1:])
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='grey', size=8)

    for label in labels:
        values = df[label].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=label)
        ax.fill(angles, values, alpha=0.25)

    plt.title(title, size=16, color='grey', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
create_radar_chart(messi_skills, 'Lionel Messi Skills', ax)
plt.show()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
create_radar_chart(lewandowski_skills, 'Robert Lewandowski Skills', ax)
plt.show()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
create_radar_chart(ronaldo_skills, 'Cristiano Ronaldo Skills', ax)
plt.show()


```

</details>

![Top 30 Most Expensive Football Players in the World](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%2030%20Most%20Expensive%20Football%20Players%20in%20the%20World.png "Top 30 Most Expensive Football Players in the World")


### Comparison of Player Skills
<details>
<summary>Click to show code</summary>

```r
# Prepare the radar chart
angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Set up radar chart
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(skills)

# Plot player stats on the radar chart
for stats, player_name in zip([player1_stats, player2_stats], ['L. Messi', 'Cristiano Ronaldo']):
    stats += stats[:1]  # Close the loop
    ax.plot(angles, stats, linewidth=1, linestyle='solid', label=player_name)
    ax.fill(angles, stats, alpha=0.1)

# Title and legend
plt.title('Comparison of Player Skills')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Show plot
plt.show()

```

</details>

![Comparison of Player Skills](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Comparison%20of%20Player%20Skills.png "Comparison of Player Skills")


### Distribution of Player Positions
<details>
<summary>Click to show code</summary>

```r
# Convert all values in 'player_positions' to strings (if not already)
df['player_positions'] = df['player_positions'].astype(str)

# Handle NaN or placeholder strings like 'nan'
df['player_positions'] = df['player_positions'].replace('nan', None)

# Split player positions
df['player_positions'] = df['player_positions'].str.split(', ')

# Create a new DataFrame with players and their different positions
positions_df = df.explode('player_positions')

# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create a frequency plot for positions
sns.countplot(data=positions_df, y='player_positions', 
              order=positions_df['player_positions'].value_counts().index)
plt.title('Distribution of Player Positions')
plt.xlabel('Number of Players')
plt.ylabel('Positions')
plt.grid(True)

# Show the plot
plt.show()

```

</details>

![Distribution of Player Positions](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Distribution%20of%20Player%20Positions.png "Distribution of Player Positions")


### Factors Leading to Goals
<details>
<summary>Click to show code</summary>

```r
# Select metrics related to scoring
columns_of_interest = [
    'short_name', 'attacking_finishing', 'power_shot_power', 'power_long_shots', 
    'attacking_volleys', 'mentality_penalties', 'mentality_composure', 'attacking_heading_accuracy'
]

# Filter data
df_analysis = df[columns_of_interest]

# Check if there are any missing values
df_analysis = df_analysis.dropna()

# Create visual plots
plt.figure(figsize=(15, 10))

# Create scatter plots for each metric versus finishing ability
sns.pairplot(df_analysis, x_vars=[
    'attacking_finishing', 'power_shot_power', 'power_long_shots', 
    'attacking_volleys', 'mentality_penalties', 'mentality_composure', 'attacking_heading_accuracy'
], y_vars='attacking_finishing', height=5, aspect=0.7, kind='reg')

plt.suptitle('Factors Leading to Goals', y=1.02)
plt.show()

```

</details>

![Factors Leading to Goals](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Factors%20Leading%20to%20Goals.png "Factors Leading to Goals")


### Factors Leading to Goals (FacetGrid)
<details>
<summary>Click to show code</summary>

```r
# Select metrics related to scoring
columns_of_interest = [
    'short_name', 'attacking_finishing', 'power_shot_power', 'power_long_shots', 
    'attacking_volleys', 'mentality_penalties', 'mentality_composure', 'attacking_heading_accuracy'
]

# Filter data
df_analysis = df[columns_of_interest]

# Select the top 20 players with the highest finishing ability
top_players = df_analysis.nlargest(20, 'attacking_finishing')

# Check if there are any missing values
top_players = top_players.dropna()

# Create FacetGrid to place the plots horizontally
g = sns.FacetGrid(
    data=top_players.melt(id_vars=['short_name', 'attacking_finishing'], 
                          value_vars=columns_of_interest[2:]),  # Columns to plot
    col="variable", 
    col_wrap=4,  # Number of columns per row
    height=4, 
    sharex=False, 
    sharey=False
)

# Create scatterplot for each plot
g.map_dataframe(sns.scatterplot, x="value", y="attacking_finishing", hue="short_name", palette="viridis", s=100)

# Add titles and alignment
g.set_titles("{col_name}")
g.set_axis_labels("Skill Value", "Attacking Finishing")
g.fig.suptitle("Factors Leading to Goals", y=1.05)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

```

</details>

![Factors Leading to Goals](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Factors%20Leading%20to%20Goals.png "Factors Leading to Goals")

### Preferred Foot and Power Shot Power, Attacking Finishing
<details>
<summary>Click to show code</summary>

```r
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Chart 1: preferred_foot and power_shot_power
sns.boxplot(
    ax=axes[0],
    x='preferred_foot',
    y='power_shot_power',
    data=df,
    hue='preferred_foot',  # Assign hue to avoid warning
    palette=['orangered', 'steelblue'],
    legend=False  # Disable legend
)
axes[0].set_title('Preferred Foot vs Power Shot Power')

# Chart 2: preferred_foot and attacking_finishing
sns.boxplot(
    ax=axes[1],
    x='preferred_foot',
    y='attacking_finishing',
    data=df,
    hue='preferred_foot',  # Assign hue to avoid warning
    palette=['orangered', 'steelblue'],
    legend=False  # Disable legend
)
axes[1].set_title('Preferred Foot vs Attacking Finishing')

plt.tight_layout()
plt.show()


```

</details>

![Preferred Foot and Power Shot Power, Attacking Finishing](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Preferred%20Foot%20and%20Power%20Shot%20Power%2C%20Attacking%20Finishing.png "Preferred Foot and Power Shot Power, Attacking Finishing")


### Manchester City Players' Stats
<details>
<summary>Click to show code</summary>

```r
mancity_df = df[df['club_name'] == "Manchester City"]

# Select the columns Name, Overall, Potential and sort by Overall in descending order
mancity_top10 = mancity_df[['short_name', 'overall', 'potential']].sort_values(by='overall', ascending=False).head(10)

# Reshape data from wide to long format
mancity_long = pd.melt(mancity_top10, id_vars=['short_name'], value_vars=['overall', 'potential'], var_name='variable', value_name='Exp')

# Create bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x='short_name', y='Exp', hue='variable', data=mancity_long, palette=["#DA291C", "#004170"])
plt.title('Manchester City')
plt.xlabel(None)
plt.ylabel('Rating')
plt.legend(title=None, loc='lower center', ncol=2)
plt.xticks(rotation=45)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()



```

</details>

### Paris Saint-Germain Players' Stats
<details>
<summary>Click to show code</summary>

```r
# Filter data for Paris Saint-Germain
psg_df = df[df['club_name'] == "Paris Saint-Germain"]

# Select the columns Name, Overall, Potential and sort by Overall in descending order
psg_top10 = psg_df[['short_name', 'overall', 'potential']].sort_values(by='overall', ascending=False).head(10)

# Reshape data from wide to long format
psg_long = pd.melt(psg_top10, id_vars=['short_name'], value_vars=['overall', 'potential'], var_name='variable', value_name='Exp')

# Create bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x='short_name', y='Exp', hue='variable', data=psg_long, palette=["#DA291C", "#004170"])
plt.title('Paris Saint-Germain')
plt.xlabel(None)
plt.ylabel('Rating')
plt.legend(title=None, loc='lower center', ncol=2)
plt.xticks(rotation=45)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
```

</details>

### Comparison of Overall and Potential by Age across Leagues
<details>
<summary>Click to show code</summary>

```r
# Calculate the average value of Overall and Potential by League and Age
df_grouped = df.groupby(['league_name', 'age']).agg({'overall': 'mean', 'potential': 'mean'}).reset_index()

# Create line plot
g = sns.FacetGrid(df_grouped, col="league_name", col_wrap=4, height=4, aspect=1.5)
g.map(sns.lineplot, "age", "overall", label="Overall", color="red", alpha=0.5)
g.map(sns.lineplot, "age", "potential", label="Potential", color="blue")

# Customize plot
for ax in g.axes.flat:
    ax.legend(loc='lower center', ncol=2)

g.add_legend(title=None, label_order=["Overall", "Potential"])
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Comparison of Overall and Potential by Age across Leagues', fontsize=16)

plt.show()
```

</details>

![Comparison of Overall and Potential by Age across Leagues](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Comparison%20of%20Overall%20and%20Potential%20by%20Age%20across%20Leagues.png "Comparison of Overall and Potential by Age across Leagues")


### Comparison of Paris Saint-Germain and Manchester City
<details>
<summary>Click to show code</summary>

```r
psg_top10['club_name'] = 'Paris Saint-Germain'
mancity_top10['club_name'] = 'Manchester City'

# Combine the data of the two clubs
combined_df = pd.concat([psg_top10, mancity_top10])

# Reshape data from wide to long format
combined_long = pd.melt(combined_df, id_vars=['short_name', 'club_name'], value_vars=['overall', 'potential'], var_name='variable', value_name='Exp')

# Create bar plot
plt.figure(figsize=(16, 10))
sns.barplot(x='short_name', y='Exp', hue='variable', data=combined_long, palette=["#DA291C", "#004170"])
plt.title('Comparison of Paris Saint-Germain and Manchester City')
plt.xlabel(None)
plt.ylabel('Rating')
plt.legend(title=None, loc='lower center', ncol=2)
plt.xticks(rotation=45)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
```

</details>

![Comparison of Paris Saint-Germain and Manchester City](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Comparison%20of%20Paris%20Saint-Germain%20and%20Manchester%20City.png "Comparison of Paris Saint-Germain and Manchester City")


### Random 10 Clubs by Contract Count
<details>
<summary>Click to show code</summary>

```r
# Get a list of random clubs
random_clubs = df['club_name'].drop_duplicates().sample(10, random_state=1).tolist()

# Filter data to include only the selected random clubs
random_club_data = df[df['club_name'].isin(random_clubs)]

# Create a column to count the number of contracts by club
contract_counts = random_club_data['club_name'].value_counts().reset_index()
contract_counts.columns = ['club_name', 'contract_count']

# Plot the chart
plt.figure(figsize=(14, 8))
sns.barplot(data=contract_counts, x='contract_count', y='club_name', hue='club_name', dodge=False, palette='viridis', legend=False)

plt.title('Random 10 Clubs by Contract Count', fontsize=16, color='darkgreen')
plt.xlabel('Count', fontsize=12, color='darkgreen')
plt.ylabel('Club Name', fontsize=12, color='darkgreen')

# Display contract count values on top of each bar
for index, value in enumerate(contract_counts['contract_count']):
    plt.text(value, index, str(value), color='black', ha="left", va="center")

plt.show()
```

</details>

![Random 10 Clubs by Contract Count](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Random%2010%20Clubs%20by%20Contract%20Count.png "Random 10 Clubs by Contract Count")


### Average Potential and Overall by Age for Bundesliga, Ligue 1, and EPL
<details>
<summary>Click to show code</summary>

```r
# Calculate the average potential and overall rating by age for each league
age_comparison = filtered_df.groupby(['league_name', 'age'])[['overall', 'potential']].mean().reset_index()

# Plot the data
plt.figure(figsize=(16, 10))
sns.lineplot(data=age_comparison, x='age', y='overall', hue='league_name', marker='o', style='league_name')
sns.lineplot(data=age_comparison, x='age', y='potential', hue='league_name', marker='D', linestyle='--', style='league_name')

plt.title('Average Potential and Overall by Age for Bundesliga, Ligue 1, and EPL', fontsize=16, color='darkblue')
plt.xlabel('Age', fontsize=12, color='darkblue')
plt.ylabel('Average Rating', fontsize=12, color='darkblue')
plt.legend(title='League / Metric')
plt.grid(True)

plt.show()

```

</details>

![Random 10 Clubs by Contract Count](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Random%2010%20Clubs%20by%20Contract%20Count.png "Random 10 Clubs by Contract Count")


### Contract Count for Bundesliga, Ligue 1, and EPL
<details>
<summary>Click to show code</summary>

```r
# Calculate the number of expiring contracts by year for each league
contract_expiry = filtered_df.groupby(['league_name', 'club_contract_valid_until']).size().reset_index(name='contract_count')

# Create plot
plt.figure(figsize=(14, 8))
sns.barplot(data=contract_expiry, x='club_contract_valid_until', y='contract_count', hue='league_name', palette='viridis')

plt.title('Number of Contracts Expiring in Each League by Year', fontsize=16, color='darkgreen')
plt.xlabel('Year', fontsize=12, color='darkgreen')
plt.ylabel('Contract Count', fontsize=12, color='darkgreen')
plt.legend(title='League')
plt.grid(True)

plt.show()


```

</details>

![Number of Contracts Expiring in Each League by Year](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Number%20of%20Contracts%20Expiring%20in%20Each%20League%20by%20Year.png "Number of Contracts Expiring in Each League by Year")


### Team Power for Every Position Class
<details>
<summary>Click to show code</summary>

```r
# Calculate the average overall rating for each club and sort in descending order
powerful = df.groupby('club_name').agg(mean_overall=('overall', 'mean')).reset_index().sort_values(by='mean_overall', ascending=False).head(20)

# Calculate the average overall rating for each club and each position class
class_mean = df.groupby(['club_name', 'Class']).agg(mean_overall=('overall', 'mean')).reset_index()

# Filter data to include only the strongest clubs
class_mean_filtered = class_mean[class_mean['club_name'].isin(powerful['club_name'])]

# Create the plot
plt.figure(figsize=(12, 8))
sns.barplot(data=class_mean_filtered, x='mean_overall', y='club_name', hue='Class', dodge=True, palette='viridis')

plt.title('Team Power for Every Position Class', fontsize=16)
plt.xlabel('')
plt.ylabel('')
plt.legend(title='Class', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)  # Adjust legend position
plt.grid(True)

plt.show()


```

</details>

![Team Power for Every Position Class](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Team%20Power%20for%20Every%20Position%20Class.png "Team Power for Every Position Class")


### Team Power for Every Position Class
<details>
<summary>Click to show code</summary>

```r
# Define positions
positions = ['Goal Keeper', 'Defender', 'Midfielder', 'Forward']

# Find the player with the highest rating in each position
top_players = df.loc[df.groupby('Class')['overall'].idxmax()]

# Convert skill columns to numeric
df[skills] = df[skills].apply(pd.to_numeric, errors='coerce')

# Extract top 4 skills for each player and create a separate table
for index, row in top_players.iterrows():
    player_skills = row[skills].astype(float).nlargest(4).reset_index()
    player_skills.columns = ['Skill', 'Value']
    player_skills['Player'] = row['long_name']
    player_skills['Class'] = row['Class']
    player_skills['Position'] = row['Class']  # Add column for position
    
    # Remove row index by resetting it
    player_skills.reset_index(drop=True, inplace=True)
    
    display(player_skills[['Player', 'Position', 'Skill', 'Value']])


```

</details>

![Team Power for Every Position Class](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Team%20Power%20for%20Every%20Position%20Class.png "Team Power for Every Position Class")


### Distribution of the Position Class in every League
<details>
<summary>Click to show code</summary>

```r
# Calculate the distribution of positions in each league
position_distribution = df.groupby(['league_name', 'Class']).size().reset_index(name='count')

# Create the plot
plt.figure(figsize=(15, 10))
sns.barplot(data=position_distribution, x='count', y='league_name', hue='Class', dodge=True, palette='viridis')

plt.title('Distribution of the Position Class in every League', fontsize=16)
plt.xlabel('Number of Players', fontsize=12)
plt.ylabel('League', fontsize=12)
plt.legend(title='Position Class', loc='upper right')
plt.grid(True)

plt.show()

```

</details>

![Average Summary Statistics of Players by Position Class in the EPL](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Average%20Summary%20Statistics%20of%20Players%20by%20Position%20Class%20in%20the%20EPL.png "Average Summary Statistics of Players by Position Class in the EPL")


### Average Summary Statistics of Players by Position Class in the EPL
<details>
<summary>Click to show code</summary>

```r
premier_league_df = df[df['league_name'] == 'English Premier League']

# Calculate the average ratings for each position in the Premier League
average_stats = premier_league_df.groupby('Class')[skills].mean().reset_index()

# Transform DataFrame from wide to long format
average_stats_long = pd.melt(average_stats, id_vars=['Class'], value_vars=skills, var_name='Skill', value_name='Average')

# Create the plot
plt.figure(figsize=(15, 10))
sns.barplot(data=average_stats_long, x='Average', y='Skill', hue='Class', dodge=True, palette='viridis')

plt.title('Average Summary Statistics of Players by Position Class in the Premier League', fontsize=16)
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('Skill', fontsize=12)
plt.legend(title='Position Class', loc='upper right')
plt.grid(True)

plt.show()


```

</details>

![Average Summary Statistics of Players by Position Class in the EPL](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Average%20Summary%20Statistics%20of%20Players%20by%20Position%20Class%20in%20the%20EPL.png "Average Summary Statistics of Players by Position Class in the EPL")


### Highest Paid Player in Each League
<details>
<summary>Click to show code</summary>

```r
# Set the font to DejaVu Sans to support the necessary characters
plt.rcParams['font.family'] = 'DejaVu Sans'

# Filter data to include only the required leagues with accurate names
leagues_p = [
    'English Premier League', 
    'French Ligue 1', 
    'Italian Serie A', 
    'Spanish Segunda División', 
    'German 1. Bundesliga', 
    'Turkish Süper Lig', 
    'Portuguese Liga ZON SAGRES'
]
filtered_df = df[df['league_name'].isin(leagues_p)]

# Convert the wage_eur column to numeric
filtered_df.loc[:, 'wage_eur'] = pd.to_numeric(filtered_df['wage_eur'], errors='coerce')

# Find the highest-paid player in each league
highest_paid_players = filtered_df.loc[filtered_df.groupby('league_name')['wage_eur'].idxmax()]

# Use the long_name column for player names
name_column = 'long_name'

# Create the plot
plt.figure(figsize=(12, 8))
sns.barplot(data=highest_paid_players, x='wage_eur', y='league_name', hue=name_column, dodge=False, palette='viridis')

plt.title('Highest Paid Player in Each League', fontsize=16)
plt.xlabel('Wage (in EUR)', fontsize=12)
plt.ylabel('League', fontsize=12)
plt.legend(title='Player')
plt.grid(True)

plt.show()



```

</details>

![Highest Paid Player in Each League](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Highest%20Paid%20Player%20in%20Each%20League.png "Highest Paid Player in Each League")


### K-Means
<details>
<summary>Click to show code</summary>

```r
# Filter forward players based on positions
forward_players = df[df['player_positions'].apply(lambda x: any(pos in forward for pos in x))]

# Skills to analyze for forward players
skills_forward = [
    'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 
    'attacking_volleys', 'skill_dribbling', 'skill_fk_accuracy', 'skill_ball_control', 
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 
    'power_shot_power', 'power_stamina', 'power_strength'
]

# Standardize skill data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(forward_players[skills_forward])

# Use PCA to reduce dimensions to 2 for plotting
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Apply K-Means algorithm with k values from 2 to 5
k_values = [2, 3, 4, 5]
clusters = {}
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=25)
    clusters[k] = kmeans.fit_predict(pca_features)

# Display clustering results with plots
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

for ax, k in zip(axes.flatten(), k_values):
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters[k], palette='viridis', ax=ax)
    ax.set_title(f'k = {k}')
    ax.set_xlabel('PCA Feature 1')
    ax.set_ylabel('PCA Feature 2')

plt.tight_layout()
plt.show()
```

</details>

![K-Means](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/K-Means.png "K-Means")

### Top 3 Wonderkids per Age Group
<details>
<summary>Click to show code</summary>

```r
# Filter players under 21 years old
wonderkids = df[df['age'] < 21]

# Sort players by potential rating
sorted_wonderkids = wonderkids.sort_values(by='potential', ascending=False)

# Select the top 3 players for each age
top_3_per_age = sorted_wonderkids.groupby('age').head(3)

# Display information of the top 3 wonderkids by age
print(top_3_per_age[['short_name', 'long_name', 'age', 'overall', 'potential', 'club_name', 'nationality_name']])

# Create a bar plot to visualize
plt.figure(figsize=(12, 8))
sns.barplot(data=top_3_per_age, x='age', y='potential', hue='short_name', dodge=True)
plt.title('Top 3 Wonderkids per Age Group', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Potential', fontsize=12)
plt.legend(title='Player', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()



```

</details>

![Top 3 Wonderkids per Age Group](https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%203%20Wonderkids%20per%20Age%20Group.png "Top 3 Wonderkids per Age Group")

