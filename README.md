# Python Project: FIFA 22 Analysis
<p align="center">
  <img src="https://www.fifaultimateteam.it/en/wp-content/uploads/2021/06/cover-fifa-22.jpg" />
</p>

- [Description](https://github.com/Tbangh16/FIFA-Analyze/edit/master/README.md#analysis-tasks)
- [Packages](#packages)
- [Prepare Data for Exploration](#prepare-data-for-exploration)
- [Process Data from Dirty to Clean](#process-data-from-dirty-to-clean)
- [Analysis Tasks](#analysis-tasks)


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
   - **`PCA (Principal Component Analysis)`**: Reduces data dimensionality and extracts key components.  
   - **`StandardScaler`**: Standardizes data to a uniform scale for better model performance.  
   - **`KMeans`**: A clustering algorithm to classify data into distinct groups.

5. **Natural Language Processing:**  
   - **`nltk`**: Offers tools for analyzing and processing text, particularly useful for handling descriptive or textual data.

6. **Mathematical Tools:**  
   - **`math (pi)`**: Provides the mathematical constant π (pi) and various functions such as trigonometry, logarithms, and power calculations, useful for geometry, angles, and advanced computations.
<details>
<summary>Click to show code</summary> 
  
```r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn import datasets
import geopandas as gpd
import nltk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import pi

```

</details>


## Prepare Data for Exploration
The <a href="https://github.com/Tbangh16/FIFA-Analyze/blob/master/players_22.csv">dataset</a> players_22.csv includes over 100 columns with detailed information on player attributes, positions, clubs, and countries.

<details>
<summary>Click to show code</summary>

```r
# Data Import
df <- read.csv("players_22.csv", encoding = "UTF-8")[-1]
head(df)
```
</details>


## Process Data from Dirty to Clean

<details>
<summary>Click to show code</summary> 
  
```r
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

<h2 align="center">Unique Values Count</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Count%20of%20Unique%20Values%20per%20Column.png" alt="Unique Values Count" width="600">
</p>

### Step 1 : Create Mapping Dictionary For Leagues And Countries

<details>
<summary>Click to show code</summary>

```r
# Define club lists for each league
bundesliga = [
  "1. FC Nürnberg", "1. FSV Mainz 05", "Bayer 04 Leverkusen", "FC Bayern München",
  "Borussia Dortmund", "Borussia Mönchengladbach", "Eintracht Frankfurt",
  "FC Augsburg", "FC Schalke 04", "Fortuna Düsseldorf", "Hannover 96",
  "Hertha BSC", "RB Leipzig", "SC Freiburg", "TSG 1899 Hoffenheim",
  "VfB Stuttgart", "VfL Wolfsburg", "SV Werder Bremen"
]

premierLeague = [
  "Arsenal", "Bournemouth", "Brighton & Hove Albion", "Burnley",
  "Cardiff City", "Chelsea", "Crystal Palace", "Everton", "Fulham",
  "Huddersfield Town", "Leicester City", "Liverpool", "Manchester City",
  "Manchester United", "Newcastle United", "Southampton", 
  "Tottenham Hotspur", "Watford", "West Ham United", "Wolverhampton Wanderers"
]

laliga = [
  "Athletic Club de Bilbao", "Atlético Madrid", "CD Leganés",
  "Deportivo Alavés", "FC Barcelona", "Getafe CF", "Girona FC", 
  "Levante UD", "Rayo Vallecano", "RC Celta", "RCD Espanyol", 
  "Real Betis", "Real Madrid", "Real Sociedad", "Real Valladolid CF",
  "SD Eibar", "SD Huesca", "Sevilla FC", "Valencia CF", "Villarreal CF"
]

seriea = [
  "Atalanta","Bologna","Cagliari","Chievo Verona","Empoli", "Fiorentina","Frosinone","Genoa",
  "Inter","Juventus","Lazio","Milan","Napoli","Parma","Roma","Sampdoria","Sassuolo","SPAL",
  "Torino","Udinese"
]

superlig = [
  "Akhisar Belediyespor","Alanyaspor", "Antalyaspor","Medipol Başakşehir FK","BB Erzurumspor","Beşiktaş JK",
  "Bursaspor","Çaykur Rizespor","Fenerbahçe SK", "Galatasaray SK","Göztepe SK","Kasimpaşa SK",
  "Kayserispor","Atiker Konyaspor","MKE Ankaragücü", "Sivasspor","Trabzonspor","Yeni Malatyaspor"
]

ligue1 = [
  "Amiens SC", "Angers SCO", "AS Monaco", "AS Saint-Étienne", "Dijon FCO", "En Avant de Guingamp",
  "FC Nantes", "FC Girondins de Bordeaux", "LOSC Lille", "Montpellier HSC", "Nîmes Olympique", 
  "OGC Nice", "Olympique Lyonnais","Olympique de Marseille", "Paris Saint-Germain", 
  "RC Strasbourg Alsace", "Stade Malherbe Caen", "Stade de Reims", "Stade Rennais FC", "Toulouse Football Club"
]

eredivisie = [
  "ADO Den Haag","Ajax", "AZ Alkmaar", "De Graafschap","Excelsior","FC Emmen","FC Groningen",
  "FC Utrecht", "Feyenoord","Fortuna Sittard", "Heracles Almelo","NAC Breda",
  "PEC Zwolle", "PSV","SC Heerenveen","Vitesse","VVV-Venlo","Willem II"
]

liganos = [
  "Os Belenenses", "Boavista FC", "CD Feirense", "CD Tondela", "CD Aves", "FC Porto",
  "CD Nacional", "GD Chaves", "Clube Sport Marítimo", "Moreirense FC", "Portimonense SC", "Rio Ave FC",
  "Santa Clara", "SC Braga", "SL Benfica", "Sporting CP", "Vitória Guimarães", "Vitória de Setúbal"
]

# Create mapping dictionary for leagues and countries
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

# Function to find league and country based on club name
def get_league_and_country(club_name):
    for league, (country, clubs) in league_country_mapping.items():
        if club_name in clubs:
            return league, country
    return None, None

# Apply the function to create 'League' and 'Country' columns
df['League'], df['Country'] = zip(*df['club_name'].apply(get_league_and_country))

# Filter out rows with 'League' value as None
df = df.dropna(subset=['League'])

# Display the first few rows of the DataFrame
print(df.head())

df['League'].head()


```
</details>

| Index | League          |
|-------|-----------------|
| 0     | Ligue 1         |
| 1     | Bundesliga      |
| 2     | Premier League  |
| 3     | Ligue 1         |
| 4     | Premier League  |

### Step 2 : Processing Currency Unit Columns
<details>
<summary>Click to show code</summary>

```r
# Check the data types of the relevant columns
print(df[['value_eur', 'wage_eur']].dtypes)

# Convert 'value_eur' and 'wage_eur' columns to strings if necessary
df['value_eur'] = df['value_eur'].astype(str)
df['wage_eur'] = df['wage_eur'].astype(str)

# Process the 'Value' and 'Wage' columns
df['Values'] = df['value_eur'].str.replace('€', '').str.replace('K', '000').str.replace('M', '').astype(float)
df['Wages'] = df['wage_eur'].str.replace('€', '').str.replace('K', '000').astype(float)

# Convert 'Values' from millions to euros
df['Values'] = df['Values'].apply(lambda x: x * 1000000 if x < 1000 else x)

# Display the first few rows of the DataFrame for checking
print(df[['value_eur', 'Values', 'wage_eur', 'Wages']].head())

```

</details>

| value_eur    | Values        | wage_eur    | Wages        |
|---------------|---------------|-------------|--------------|
| 78000000.0    | 78000000.0    | 320000.0    | 320000.0     |
| 119500000.0   | 119500000.0   | 270000.0    | 270000.0     |
| 45000000.0    | 45000000.0    | 270000.0    | 270000.0     |
| 129000000.0   | 129000000.0   | 270000.0    | 270000.0     |
| 125500000.0   | 125500000.0   | 350000.0    | 350000.0     |

### Step 3 : Classify Players And Preferred Foot

<details>
<summary>Click to show code</summary>

```r
# Filter values "Left" and "Right"
df = df[df['preferred_foot'].isin(["Left", "Right"])]

defence = ["CB", "RB", "LB", "LWB", "RWB", "LCB", "RCB"]
midfielder = ["CM", "CDM","CAM","LM","RM", "LAM", "RAM", "LCM", "RCM", "LDM", "RDM"]
forward = ["CF", "ST", "LW", "RW", "LS", "RS", "LF", "RF"]

# Classify players
df['Class'] = df['club_position'].apply(lambda x: 'Goal Keeper' if x == "GK" else
                                                  'Defender' if x in defence else
                                                  'Midfielder' if x in midfielder else
                                                  'Forward' if x in forward else
                                                  'Unknown')

# Display the first few rows of the DataFrame for checking
print(df[['club_position', 'Class', 'Preferred.Foot']].head())
```

</details>

| club_position | Class        | Preferred.Foot |
|----------------|--------------|----------------|
| RW             | Forward      | Left           |
| ST             | Forward      | Right          |
| ST             | Forward      | Right          |
| LW             | Forward      | Right          |
| RCM            | Midfielder   | Right          |



### Step 4 : Remove Unnecessary Data Columns And Check

<details>
<summary>Click to show code</summary>

```r
# Drop unnecessary columns
df = df.drop(columns=[
    "sofifa_id", "body_type", "real_face", "club_joined", "club_loaned_from",
    "release_clause_eur", "player_face_url", "club_flag_url", "club_logo_url", "nation_flag_url", 
    "work_rate"
])

# Plot the count of unique values per column
plt.figure(figsize=(14, 10))
colors = plt.cm.Blues(np.linspace(0.3, 1, len(df.nunique())))
df.nunique().sort_values(ascending=False).plot(kind='bar', color=colors)
plt.title('Count of Unique Values per Column', fontsize=12, color='darkgreen')
plt.show()
```
</details>

<h2 align="center">Unique Values Count</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Count%20of%20Unique%20Values%20per%20Column%201.png" alt="Unique Values Count" width="600">
</p>

# Analysis Tasks


## Distribution & The Average Age of The Players in each League

* **Ligue 1**: Average age is 24.12, youthful talent.
* **Bundesliga**: Average age is 24.13, similar to Ligue 1.
* **Premier League**: Average age is 24.85, balancing youth and experience.
* Eredivisie: Youngest average age at 23.1.
* Serie A: Oldest average age at 26.17.
* Other leagues range between 23.97 and 26.05 years old.

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

<h2 align="center">Distribution & The Average Age of The Players in Each League</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Distribution%20%26%20The%20Average%20Age%20of%20The%20Players%20in%20each%20League.png" alt="Distribution & The Average Age of The Players in each League" width="800">
</p>


## Average, Oldest, and Youngest Age of Players by Country

*   **Average Age:** Most teams have an average age between 24 and 26.
*   **Oldest Players:** Gianluigi Buffon (43, Italy) is the oldest player. Several others are in their late 30s (e.g., N. Penneteau, Bracali, M. Stekelenburg, A. Hutchinson, B. Foster, M. Hasebe, Riesgo).
*   **Youngest Players:** Many players are 16 or 17 years old, indicating a focus on youth development (e.g., R. Cherki, W. Faghir, K. Urbański, R. van den Berg, Tiago Morais, Gavi, E. Bilgin, T. Small).

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

<h2 align="center">Average, Oldest, and Youngest Age of Players by Country</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Average%2C%20Oldest%2C%20and%20Youngest%20Age%20of%20Players%20by%20Country.png" alt="Average, Oldest, and Youngest Age of Players by Country" width="800">
</p>



## Market Values of the Leagues


* **Highest Market Value:** The English Premier League exceeds 8 billion €.
* **Lowest Market Value:** Italian Serie B is just above 0 billion €.
* **Top Five Leagues:** Dominated by the English Premier League, German 1. Bundesliga, Italian Serie A, French Ligue 1, and Spain Primera Division.
* **Significant Disparity:** A wide range in market values, indicating economic disparities.
* **Concentration of Talent:** Higher market values suggest a concentration of talent and financial resources in top leagues.

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
<h2 align="center">Market Values of the Leagues</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Market%20Values%20of%20the%20Leagues.png" alt="Market Values of the Leagues" width="800">
</p>


## Top 3 Teams with Highest Value in Each League

* **Highest Market Value:** Paris Saint-Germain.
* **Leagues Representation:** Top teams from various leagues like French Ligue 1, German 1. Bundesliga, English Premier League, and more.
* **Wide Range:** Market values vary significantly among the teams.
* **Top Teams by Leagues:** Dominance of teams from top European leagues.
* **Economic Disparity:** Noticeable gap in market values across different teams and leagues.


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
<h2 align="center">Top 3 Teams with Highest Value in Each League</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%203%20Teams%20with%20Highest%20Value%20in%20Each%20League.png" alt="Top 3 Teams with Highest Value in Each League" width="800">
</p>



## Top 30 Most Expensive Football Players in the World

* **Kylian Mbappé tops the list as the most expensive player, significantly ahead of others.**  
* **The chart highlights young talents like Haaland and Sancho alongside experienced players like Lewandowski.**  
* **Paris Saint-Germain dominates with multiple players featured, reflecting its strong financial backing.**  
* **Premier League clubs like Manchester City and Liverpool have a substantial presence on the list.**  


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
<h2 align="center">Top 30 Most Expensive Football Players in the World</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%2030%20Most%20Expensive%20Football%20Players%20in%20the%20World.png" alt="Top 30 Most Expensive Football Players in the World" width="800">
</p>



## Top 20 players with the highest Skill

* **Lionel Messi leads the chart with the highest skill rating, closely followed by Lewandowski and Ronaldo.**  
* **The chart reflects a balance between forwards, midfielders, and goalkeepers in the top skill rankings.**  
* **Notable defenders like Sergio Ramos and Van Dijk are included, highlighting their technical abilities.**  
* **Younger talents such as Kylian Mbappé stand out alongside seasoned players like Neuer and Messi.**  


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

<h2 align="center">Top 20 Players with the Highest Skill</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%2020%20players%20with%20the%20highest%20Skill.png" alt="Top 20 players with the highest Skill" width="800">
</p>



## Comparison of Messi, Ronaldo, and Lewandowski

* **Lionel Messi excels in dribbling and vision, showcasing his playmaking skills.**
* **Cristiano Ronaldo displays outstanding shot power and curve, emphasizing his goal-scoring prowess.**
* **Robert Lewandowski has a balanced skill set with notable strengths in balance and shot power.**
* **The charts highlight the unique strengths of each player across various skill attributes.**
* **Overall, the visual comparison effectively illustrates the diverse skill sets of these top footballers.**


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

<h2 align="center">Comparison of Messi, Ronaldo, and Lewandowski</h2>
<p align="center">
  <img src="https://github.com/Tbangh16/FIFA-Analyze/blob/master/photo/lionel%20messi.png" alt="Lionel Messi" width="300" height="300">
  <img src="https://github.com/Tbangh16/FIFA-Analyze/blob/master/photo/ronaldo.png" alt="Cristiano Ronaldo" width="300" height="300">
  <img src="https://github.com/Tbangh16/FIFA-Analyze/blob/master/photo/lewandowski.png" alt="Robert Lewandowski" width="300" height="300">
</p>




## Comparison of Player Skills

* **Lionel Messi has high ratings in dribbling and passing, showcasing his playmaking ability.**
* **Cristiano Ronaldo excels in shooting and physic, emphasizing his goal-scoring prowess.**
* **The radar chart highlights the contrasting strengths of each player in different skill areas.**
* **Messi’s lower physic rating contrasts with Ronaldo’s strength in this area.**
* **Overall, the chart effectively visualizes the unique skills of both players.**


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

<h2 align="center">Comparison of Player Skills (Messi vs Ronaldo)</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Comparison%20of%20Player%20Skills(%20Radar).png" alt="Radar Chart" width="500">
</p>


## Distribution of Player Positions

* **Central Midfielders (CM) have the highest number of players, totaling around 750.**
* **Center Backs (CB) follow with approximately 700 players, showcasing the importance of defense.**
* **Strikers (ST) have a significant presence, with around 600 players listed.**
* **Central Defensive Midfielders (CDM) and Central Attacking Midfielders (CAM) also show strong representation with 550 and 500 players, respectively.**
* **The chart highlights the distribution of players across different positions, emphasizing the depth in midfield and defense.**


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

<h2 align="center">Distribution of Player Positions</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Distribution%20of%20Player%20Positions.png" alt="Distribution of Player Positions" width="800">
</p>

## Factors Leading to Goals

* **The histogram indicates that the majority of players have a high attacking_finishing score.**
* **The scatter plot between attacking_finishing and power_shot_power shows a strong positive correlation.**
* **There is a noticeable trend between attacking_finishing and power_long_shots, indicating skill overlap.**
* **Attacking_volleys and mentality_penalties both have moderate correlations with attacking_finishing.**
* **Overall, the plots reveal various factors contributing to successful attacking plays.**


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

<h2 align="center">Factors Leading to Goals</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Factors%20Leading%20to%20Goals.png" alt="Factors Leading to Goals" heigh="500",width="800">
</p>



## Factors Leading to Goals (FacetGrid)

* **Attacking Finishing** shows a strong positive correlation with **power_shot_power** and **power_long_shots**.
* **Attacking Volleys** and **mentality_penalties** display a significant positive correlation with **Attacking Finishing**.
* **Mentality Composure** and **Attacking Heading Accuracy** have moderate correlations with **Attacking Finishing**.
* **Players like Messi and Ronaldo exhibit high values in key skills, enhancing their goal-scoring capabilities.**
* **The charts effectively visualize the relationship between various skill values and Attacking Finishing.**


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
<h2 align="center">Factors Leading to Goals (FacetGrid)</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Factors%20Leading%20to%20Goals%201.png" alt="Factors Leading to Goals" width="800">
</p>

## Preferred Foot and Power Shot Power, Attacking Finishing

* **Left-footed players have a slightly lower median Power Shot Power compared to Right-footed players.**
* **The interquartile ranges for Power Shot Power are similar for both foot preferences, spanning from 60 to 80.**
* **Attacking Finishing shows a higher median for Right-footed players (around 60) compared to Left-footed players (around 55).**
* **Both box plots highlight the distributions and central tendencies of Power Shot Power and Attacking Finishing.**
* **Outliers are present in the Power Shot Power distribution for Left-footed players, but not for Right-footed players.**


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
<h2 align="center">Preferred Foot and Power Shot Power, Attacking Finishing</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/'Preferred%20Foot%20vs%20Attacking%20Finishing.png" alt="Preferred Foot and Power Shot Power, Attacking Finishing" width="800">
</p>



## Comparison of Overall and Potential by Age across Leagues

* **Players' overall and potential ratings increase with age across various football leagues.**
* **English Premier League players show a steady rise in ratings until their mid-20s.**
* **French Ligue 1 and German 1. Bundesliga show similar trends, with potential ratings peaking around age 23-24.**
* **Spanish Primera Division players exhibit a consistent increase in overall ratings into their late 20s.**
* **The graphs highlight differences in player development and peak performance across leagues.**


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
<h2 align="center">Comparison of Overall and Potential by Age across Leagues</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Comparison%20of%20Overall%20and%20Potential%20by%20Age%20across%20Leagues.png" alt="Comparison of Overall and Potential by Age across Leagues" width="800">
</p>


## Comparison of Paris Saint-Germain and Manchester City

* **Key Players**: Notable names include Messi, Neymar Jr, and Mbappé from PSG, and De Bruyne, Sterling, and Ederson from Manchester City.
* **Rating Highlights**: Both overall and potential ratings are represented, highlighting current performance and future potential.
* **Team Strengths**: PSG's top players generally have high potential ratings, while Manchester City's players show strong current ratings.
* **Visual Comparison**: The chart visually emphasizes the strengths and potential growth of players from both clubs.



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
<h2 align="center">Comparison of Paris Saint-Germain and Manchester City</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Comparison%20of%20Paris%20Saint-Germain%20and%20Manchester%20City.png" alt="Comparison of Paris Saint-Germain and Manchester City" width="800">
</p>



## Random 10 Clubs by Contract Count

* **West Ham United** has the highest number of contracts, totaling 33.
* **1. FSV Mainz 05** follows closely with 32 contracts.
* **FC Nantes** ranks third with 31 contracts.
* The chart provides a visual comparison of the contract counts among ten football clubs.
* It highlights which clubs have more or fewer contracts, with **West Ham United** and **OGC Nice** having the highest and lowest counts, respectively.


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
<h2 align="center">Random 10 Clubs by Contract Count</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Random%2010%20Clubs%20by%20Contract%20Count.png" alt="Random 10 Clubs by Contract Count" width="800">
</p>



## Average Potential and Overall by Age for Bundesliga, Ligue 1, and EPL

* **Average Ratings:** The graph tracks average player ratings by age for the English Premier League, French Ligue 1, and German 1. Bundesliga.
* **Rating Trends:** Players in all leagues show an increase in average ratings until their mid-20s.
* **League Comparison:** The English Premier League has the highest average ratings, peaking around age 27.
* **Performance Decline:** All leagues experience a decline in average ratings after age 30.
* **Development:** The graph highlights the development and peak performance periods of players in different leagues.


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
<h2 align="center">Average Potential and Overall by Age for Bundesliga, Ligue 1, and EPL</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Average%20Potential%20and%20Overall%20by%20Age%20for%20Bundesliga%2C%20Ligue%201%2C%20and%20EPL.png" alt="Average Potential and Overall by Age for Bundesliga, Ligue 1, and EPL" width="800">
</p>



## Contract Count for Bundesliga, Ligue 1, and EPL

* **Highest Number of Expirations in 2022:** The English Premier League leads with the most contracts expiring, followed by French Ligue 1 and German 1. Bundesliga.
* **Trend Over the Years:** The number of expiring contracts declines as the years progress towards 2027.
* **Year-to-Year Comparison:** Each league shows varying counts for contract expirations over the years.
* **League Distribution:** The chart compares the distribution and timing of contract expirations in three major football leagues.
* **Notable Decline:** A noticeable decrease in the number of expiring contracts is observed from 2023 onwards.



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
<h2 align="center">Number of Contracts Expiring in Each League by Year</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Number%20of%20Contracts%20Expiring%20in%20Each%20League%20by%20Year.png" alt="Number of Contracts Expiring in Each League by Year" width="800">
</p>



## Team Power for Every Position Class
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
<h2 align="center">Team Power for Every Position Class</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Team%20Power%20for%20Every%20Position%20Class.png" alt="Team Power for Every Position Class" width="800">
</p>



## Team Power for Every Position Class

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

## Player Statistics

### Virgil van Dijk
| Player             | Position  | Skill      | Value |
|--------------------|-----------|------------|-------|
| Virgil van Dijk    | Defender  | defending  | 91.0  |
| Virgil van Dijk    | Defender  | physic     | 84.0  |
| Virgil van Dijk    | Defender  | pace       | 78.0  |
| Virgil van Dijk    | Defender  | dribbling  | 72.0  |

### Lionel Andrés Messi Cuccittini
| Player                             | Position  | Skill      | Value |
|------------------------------------|-----------|------------|-------|
| Lionel Andrés Messi Cuccittini     | Forward   | dribbling  | 95.0  |
| Lionel Andrés Messi Cuccittini     | Forward   | shooting   | 92.0  |
| Lionel Andrés Messi Cuccittini     | Forward   | passing    | 91.0  |
| Lionel Andrés Messi Cuccittini     | Forward   | pace       | 85.0  |

### Manuel Peter Neuer
| Player             | Position      | Skill      | Value |
|--------------------|---------------|------------|-------|
| Manuel Peter Neuer | Goal Keeper   | pace       | NaN   |
| Manuel Peter Neuer | Goal Keeper   | shooting   | NaN   |
| Manuel Peter Neuer | Goal Keeper   | passing    | NaN   |
| Manuel Peter Neuer | Goal Keeper   | dribbling  | NaN   |

### Kevin De Bruyne
| Player            | Position    | Skill      | Value |
|-------------------|-------------|------------|-------|
| Kevin De Bruyne   | Midfielder  | passing    | 93.0  |
| Kevin De Bruyne   | Midfielder  | dribbling  | 88.0  |
| Kevin De Bruyne   | Midfielder  | shooting   | 86.0  |
| Kevin De Bruyne   | Midfielder  | physic     | 78.0  |


## Distribution of the Position Class in every League

* **Defenders and Midfielders dominate** across most leagues, with the highest counts in the **English League Championship** and **English Premier League**.
* **Forwards and Goal Keepers** have fewer players in comparison, highlighting the specialized nature of these positions.
* The **German 1. Bundesliga** and **Italian Serie A** show balanced distributions among position classes.
* **French Ligue 1** and **Spain Primera Division** emphasize their depth in Midfielders and Defenders.
* The chart effectively illustrates team compositions and player distributions across major European soccer leagues.


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
<h2 align="center">Distribution of the Position Class in Every League</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Distribution%20of%20the%20Position%20Class%20in%20every%20League.png" alt="Distribution of the Position Class in every League" width="800">
</p>



## Average Summary Statistics of Players by Position Class in the EPL

* **Defenders and Midfielders:** Excelling in defending and physic, as expected.
* **Forwards:** Strong in shooting and dribbling, highlighting their offensive skills.
* **Goal Keepers:** Leading in physic, showcasing their physical strength and agility.
* **Midfielders:** Stand out in passing and dribbling, reflecting their playmaking abilities.
* **Overall Comparison:** The chart effectively highlights the strengths and weaknesses of different position classes in the Premier League.


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
<h2 align="center">Average Summary Statistics of Players by Position Class in the EPL</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Average%20Summary%20Statistics%20of%20Players%20by%20Position%20Class%20in%20the%20Premier%20League.png" alt="Average Summary Statistics of Players by Position Class in the EPL" width="800">
</p>



## Highest Paid Player in Each League

* **Highest Paid Player**: Kevin De Bruyne leads with the highest wage in the English Premier League at around 350,000 EUR.
* **French Ligue 1**: Lionel Messi follows closely as the top earner.
* **Other Leagues**: Robert Lewandowski (German 1. Bundesliga), Paulo Dybala (Italian Serie A), Pedro Porro (Portuguese Liga ZON SAGRES), Sergi Palencia (Spanish Segunda División), and Miralem Pjanić (Turkish Süper Lig) are highlighted.
* **Wage Range**: Wages range from approximately 50,000 EUR to 350,000 EUR.
* **Comparison**: The chart effectively visualizes the wage differences among top players across various football leagues.

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
<h2 align="center">Highest Paid Player in Each League</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Highest%20Paid%20Player%20in%20Each%20League.png" alt="Highest Paid Player in Each League" width="800">
</p>



## K-Means

* **k = 2 Clusters**: The plot shows a clear distinction between two main clusters, indicating a fundamental grouping of forward players based on their skills.
* **k = 3 Clusters**: Adding a third cluster reveals more nuanced groupings, capturing additional subgroups within the primary clusters.
* **k = 4 Clusters**: Four clusters highlight even finer distinctions among players, offering a more detailed segmentation based on skill sets.
* **k = 5 Clusters**: With five clusters, the plot provides the most detailed segmentation, although some clusters may overlap slightly.
* **Overall Insight**: These scatter plots effectively visualize how different values of k affect the clustering of forward players' skills, aiding in identifying the optimal number of clusters for detailed analysis.


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
<h2 align="center">K-Means</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/K-Means.png" alt="K-Means" width="800">
</p>


## Top 3 Wonderkids per Age Group

* **Top Potential Ratings**: The chart displays potential ratings of top young football players across different age groups (17, 18, 19, and 20 years old).
* **Age 17**: M. Cho, Pedri, and J. Bellingham are the top 3 wonderkids with high potential ratings.
* **Age 18**: F. Wirtz, R. Gravenberch, and M. Greenwood lead this age group in potential.
* **Age 19**: B. Saka, E. Haaland, and A. Davies have the highest potential ratings.
* **Age 20**: D. Szoboszlai, followed by other standout young talents, dominates this category.


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
<h2 align="center">Top 3 Wonderkids per Age Group</h2>
<p align="center">
  <img src="https://raw.githubusercontent.com/Tbangh16/FIFA-Analyze/master/photo/Top%203%20Wonderkids%20per%20Age%20Group.png" alt="Top 3 Wonderkids per Age Group" width="800">
</p>


