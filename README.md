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

  
## Source
The <a href="https://github.com/Tbangh16/FIFA-Analyze/blob/master/players_22.csv">dataset</a> players_22.csv includes over 100 columns with detailed information on player attributes, positions, clubs, and countries.
# Dữ liệu Cầu Thủ FIFA 19

Trong notebook này, chúng ta sẽ nhập và xem trước dữ liệu cầu thủ từ file CSV `data.csv`.

<details>
<summary>Click to show code</summary>

```r
# Data Import
df <- read.csv("players_22.csv", encoding = "UTF-8")[-1]
head(df)
```
```r
# Hiển thị thông tin chung về DataFrame 
print("Thông tin chung về DataFrame:") 
print(df.info()) 
# Hiển thị thống kê mô tả 
print("\nThống kê mô tả:") 
print(df.describe()) 
# Hiển thị những dòng đầu tiên của DataFrame 
print("\nNhững dòng đầu tiên của DataFrame:") 
print(df.head())
```
</details>

## Prepare Data for Exploration

## Process Data from Dirty to Clean

<details>
<summary>Click to show code</summary> 
  
```r
# Biểu đồ tỉ lệ giá trị thiếu
missing = df.isnull().mean() * 100
missing = missing[missing > 0].sort_values()
plt.figure(figsize=(10, 6))
colors = plt.cm.Reds(np.linspace(0.3, 1, len(missing)))
missing.plot(kind='bar', color=colors)
plt.title('Percentage of Missing Values', fontsize=16, color='darkred')
plt.xlabel('Columns', fontsize=12, color='darkred')
plt.ylabel('Percentage', fontsize=12, color='darkred')
plt.show()

# Biểu đồ số lượng mỗi kiểu dữ liệu
plt.figure(figsize=(10, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', 'pink', 'yellow']
df.dtypes.value_counts().plot(kind='bar', color=colors)
plt.title('Count of Data Types', fontsize=16, color='darkorange')
plt.xlabel('Data Types', fontsize=12, color='darkorange')
plt.ylabel('Count', fontsize=12, color='darkorange')
plt.show()

# Biểu đồ số lượng các giá trị duy nhất trong mỗi cột
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
