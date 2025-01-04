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
