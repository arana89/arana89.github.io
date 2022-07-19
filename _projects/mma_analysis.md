---
title: "MMA Gym Analysis"
excerpt: "Aggregation, analysis and geospatial visualization of MMA fighter data."
header:
  overlay_image: /assets/images/banner.png
  overlay_filter: 0.75
  teaser: /assets/images/mma_analysis/teaser.png
tag:
- pandas
- scikit-learn
- folium
- GIS
- GCP
- SQL
toc: true
toc_sticky: true
last_modified_at: 2022-06-01
number: 2 #for sorting
---

# Summary
Gyms play an important role in the development of the rapidly evolving meta strategy of mixed martial arts (MMA). Motivated by curiosity as a casual fan and hobbyist of the sport, I aggregate and visualize publicly available data to provide insight into gym performance and distribution across California.

# Scraping and Tabulating: Pandas and BeautifulSoup
California's amateur MMA governing body maintains a website with fighter information and bout statistics. Queries are limited to searching for individual fighters.

<figure style="width: 600px" class="align-center">
  <img src="/assets/images/mma_analysis/website_screenshot.png" alt="">
  <figcaption>The data as it is presented on the website of the governing body for amateur MMA in California.</figcaption>
</figure>

The data exist as a paginated table, making it relatively straightforward to scrape. I used the Pandas function `read_html()` to scrape and pipe the data directly into a dataframe. It uses `BeautifulSoup4` on the back-end, which itself is just a wrapper for a set of html parsers. This loop goes through each page of the table and appends the data to the main dataframe object:

```python
import pandas as pd

page_size = 20 #number of rows per page
row_count = 1
keep_going = True
df = pd.DataFrame()

while keep_going:
    url = f'https://camomma.org/Users-fighters&from={row_count}'
    temp_df = pd.read_html(url)[0]    
    df = pd.concat([df,temp_df])    
    row_count += page_size

    if len(temp_df) == 0:
        keep_going = False
```

The resulting dataframe contains the fighter's name, nickname, age, record, weight class, gym association and city:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Name</th>
      <th>Weight Class</th>
      <th>Association / Ring Name</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Derrick Ko  MMA record: 0-0-0-0 Age: 25</td>
      <td>Bantamweight - 125 to 135 pounds</td>
      <td>Goon Squad MMADerrick "Cash" Ko</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <td>Kaien Cicero  MMA record: 0-0-0-0 Age: 20</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>None</td>
      <td>Brea</td>
    </tr>
    <tr>
      <td>Darrian Williams  MMA record: 0-0-0-0 Age: 28</td>
      <td>Heavyweight - 230.1 to 265 pounds</td>
      <td>Samurai Dojo</td>
      <td>Exeter</td>
    </tr>
    <tr>
      <td>Christian Zubiate  MMA record: 0-0-0-0 Age: 25</td>
      <td>Welterweight - 155.1 to 170 pounds</td>
      <td>10th planetHead Hunter</td>
      <td>Newport Beach</td>
    </tr>
    <tr>
      <td>Kevin Wright  MMA record: 0-0-0-0 Age: 25</td>
      <td>Middleweight - 170.1 to 185 pounds</td>
      <td>10th PlanetKevin Wright</td>
      <td>phoenix</td>
    </tr>
  </tbody>
</table>
</div>

The formatting isn't quite right, as the ***Name*** column also contains the fighter's record and age. However, this can be fixed with a little bit of regex'ing:

```python
df[['name', 'record', 'age']] = df['Name'].str.extract('(?i)(?P<name>.+?(?=\sMMA))\sMMA\srecord:\s(?P<record>(?<=MMA\srecord:\s)\d+-\d+-\d+-\d+).*?\sAge:\s(?P<age>\d+)', expand=True)
df.pop('Name')

#reorder columns
cols = df.columns.to_list()
cols = cols[-3:] + cols[:-3]
df = df[cols]
```

Here is a breakdown of the expression:

```
(?i)(?P<name>.+?(?=\sMMA))\sMMA\srecord:\s(?P<record>(?<=MMA\srecord:\s)\d+-\d+-\d+-\d+).*?\sAge:\s(?P<age>\d+)

(?i) : global case insensitive modifier

(?P<name>.+?(?=\sMMA)) : lazy match everything until reaching 'MMA'

(?P<record>(?<=MMA\srecord:\s)\d+-\d+-\d+-\d+) :  After 'MMA record: ', capture the 4 digits that are separated by dashes

.*?\s : Lazy catch for rows where a pankration/combat grappling record was also listed for a fighter. This data wasn't used.

Age:\s(?P<age>\d+) : Capture the digits after 'Age: '
```

After the initial formatting, the table looks like:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    {: .align-center}

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>record</th>
      <th>age</th>
      <th>Weight Class</th>
      <th>Association / Ring Name</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Derrick Ko</td>
      <td>0-0-0-0</td>
      <td>25</td>
      <td>Bantamweight - 125 to 135 pounds</td>
      <td>Goon Squad MMADerrick "Cash" Ko</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <td>Kaien Cicero</td>
      <td>0-0-0-0</td>
      <td>20</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>None</td>
      <td>Brea</td>
    </tr>
    <tr>
      <td>Darrian Williams</td>
      <td>0-0-0-0</td>
      <td>28</td>
      <td>Heavyweight - 230.1 to 265 pounds</td>
      <td>Samurai Dojo</td>
      <td>Exeter</td>
    </tr>
    <tr>
      <td>Christian Zubiate</td>
      <td>0-0-0-0</td>
      <td>25</td>
      <td>Welterweight - 155.1 to 170 pounds</td>
      <td>10th planetHead Hunter</td>
      <td>Newport Beach</td>
    </tr>
    <tr>
      <td>Kevin Wright</td>
      <td>0-0-0-0</td>
      <td>25</td>
      <td>Middleweight - 170.1 to 185 pounds</td>
      <td>10th PlanetKevin Wright</td>
      <td>phoenix</td>
    </tr>
  </tbody>
</table>
</div>

After a bit more re-structuring and filtering, I get a table that looks like:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>age</th>
      <th>weight</th>
      <th>gym</th>
      <th>city</th>
      <th>mma_w</th>
      <th>mma_l</th>
      <th>mma_d</th>
      <th>mma_nc</th>
      <th>mma_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gabriella Sullivan</td>
      <td>26</td>
      <td>Flyweight - 115.1 to 125 pounds</td>
      <td>Azteca</td>
      <td>Lakeside</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Jeremiah Garber</td>
      <td>27</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>10th Planet HQ - The Yard</td>
      <td>Los Angeles</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Rodrigo Oliveira</td>
      <td>32</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>Triunfo Jiu-jitsu &amp; mma</td>
      <td>Pico Rivera</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Ronell White</td>
      <td>20</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>Team Quest</td>
      <td>Oceanside</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Enrique Valdez</td>
      <td>22</td>
      <td>Middleweight - 170.1 to 185 pounds</td>
      <td>Yuma top team</td>
      <td>Yuma</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

# Grouping and Aggregation
The first step to any sort of gym-wise analysis is to group the data by gym. If the data veracity is perfect, this is accomplished with a simple call to `groupby()`. However, the gym name field is a human-typed string, subject to a variety of inconsistencies including:

1. typos, e.g.
```
['systems training center', 'syatems training center']
```

2. abbreviations, e.g.
```
['american kickboxing academy', 'american kickboxing academy (aka)',
 'aka (american kickboxing academy)',
 'american kickboxing academy a.k.a', 'aka', 'a.k.a.']
```

3. prefix/suffix, e.g.
```
['bear pit', 'the bear pit', 'bear pit mma']
```

4. semantics, e.g.
```
['independent', 'no team', 'solo', 'myself']
```

The semantics issue was relatively minor, so I handled it manually. I approached the other three issues collectively as a clustering problem. This immediately gave rise to two questions. Firstly, clustering necessitates a notion of distance between elements. What is a good way to define distance between strings? Secondly, which type of clustering is appropriate here?

## A Distance Metric for Strings
There have been a wide variety of string metrics developed to address the problem of approximate string matching. Most approaches devise a score by counting the number of some well-defined operations needed to transform string A to string B. As an example, [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) counts the number of substitutions, additions and deletions needed to transform one string into another.

$$
  lev(\text{kitten},\text{sitting}) = 3 \\
  \textbf{k}\text{itten} \rightarrow \textbf{s}\text{itten} \\
  \text{sitt}\textbf{e}\text{n} \rightarrow \text{sitt}\textbf{i}\text{n} \\
  \text{sittin} \rightarrow \text{sittin} \textbf{g} \\
$$

After testing a few different metrics, [Jaro similarity](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance#Jaro_similarity) produced the most reliable groupings.

## Agglomerative Clustering
Clustering refers to the process of grouping data into subsets based on some criteria of similarity, i.e. distance, mean value, density, connectivity etc. There are many different algorithms to chose from because a cluster is not a well-defined object -- the best way to define a cluster depends on the context of the problem.  

<figure style="width: 100%" class="align-center">
  <img src="/assets/images/mma_analysis/scikit_learn_clustering_comparison.png" alt="">
  <figcaption>A comparison of various clustering algorithms. Figure reproduced from the <a href="https://scikit-learn.org/">scikit-learn</a> documentation.</figcaption>
</figure>

Hierarchical clustering aims to build nested groups of data as a function of a pairwise distance metric. This grouping permits an ordering of the clusters into levels, forming a hierarchy. Agglomerative clustering begins with each data point in its own cluster, and iteratively merges clusters based on distance and connectivity constraints. This is in contrast to divisive clustering, which is also hierarchical, but begins with all the data in a single cluster and iteratively divides them into individual clusters. The flexibility in choosing a distance metric, and not needing *a priori* knowledge about the number of clusters makes hierarchical clustering a good choice for the gym data.

Before clustering, I cleaned the list of gym names by removing elements that would dilute the metric, such as spaces and commonly used terms:

```python
gym_names_full = df['gym'].str.lower().str.replace(" ","").tolist()

import re
gym_names = [re.sub('\W', '', i) for i in gym_names_full]

#remove common string patterns to improve clustering accuracy
common_gym_terms = ['jiujitsu', 'mma', 'team', 'fighter',
                    'gym', 'fight', 'center', 'fighting']

for i, gym in enumerate(gym_names):

    for j in common_gym_terms:
        gym = gym.replace(j, '')

    gym_names[i] = gym
```

Then, I pre-computed the distance matrix and performed clustering:

```python
from sklearn import cluster
from jellyfish import jaro_distance

#pre-compute the pairwise distance matrix
X_jaro_distance = np.zeros((len(gym_names), len(gym_names)))
for ii in np.arange(len(gym_names)):
    for jj in np.arange(len(gym_names)):
        X_jaro_distance[ii][jj] = 1 - jaro_distance(str(gym_names[ii]),
                                                    str(gym_names[jj]))

clustering = cluster.AgglomerativeClustering(n_clusters=None,
                                             distance_threshold=0.2,
                                             affinity='precomputed',
                                             linkage='complete'
                                             ).fit(X_jaro_distance)
```

Note: It is important to understand the effect of the linkage criterion on the clustering result. Linkage is how the distance between sets is calculated when determining whether or not to merge them. *Complete* linkage means that we consider the maximum pairwise distance between all members in the two sets as the cluster distance. In contrast, *single* linkage uses the minimum pairwise distance between members of the two sets. Single linkage can cause runaway cluster behavior, where one cluster grows extremely large.
{: .notice--info}

<figure style="width: 100%" class="align-center">
  <img src="/assets/images/mma_analysis/gym_dendrogram.png" alt="">
  <figcaption>A dendrogram illustrating the results of agglomerative clustering by Jaro similarity on a subset of the gym names.</figcaption>
</figure>

The object returned by `AgglomerativeClustering()` includes the cluster labels for each item, which I append to my dataframe as the column `gym_cluster_labels`.

```python
df['gym_cluster_labels'] = clustering.labels_
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>age</th>
      <th>weight</th>
      <th>gym</th>
      <th>city</th>
      <th>mma_w</th>
      <th>mma_l</th>
      <th>mma_d</th>
      <th>mma_nc</th>
      <th>mma_total</th>
      <th>gym_cluster_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gabriella Sullivan</td>
      <td>26</td>
      <td>Flyweight - 115.1 to 125 pounds</td>
      <td>Azteca</td>
      <td>Lakeside</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1255</td>
    </tr>
    <tr>
      <td>Jeremiah Garber</td>
      <td>27</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>10th Planet HQ - The Yard</td>
      <td>Los Angeles</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>324</td>
    </tr>
    <tr>
      <td>Rodrigo Oliveira</td>
      <td>32</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>Triunfo Jiu-jitsu &amp; mma</td>
      <td>Pico Rivera</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>206</td>
    </tr>
    <tr>
      <td>Ronell White</td>
      <td>20</td>
      <td>Lightweight - 145.1 to 155 pounds</td>
      <td>Team Quest</td>
      <td>Oceanside</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>719</td>
    </tr>
    <tr>
      <td>Enrique Valdez</td>
      <td>22</td>
      <td>Middleweight - 170.1 to 185 pounds</td>
      <td>Yuma top team</td>
      <td>Yuma</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>785</td>
    </tr>
  </tbody>
</table>
</div>

Now with reliable labels, it becomes possible to aggregate and get descriptive statistics. A natural question may be: how successful are the largest gyms? This code returns `g` as a dataframe containing the name, total mma fight count and win rate of each gym:

```python
g = df.groupby(['gym_cluster_labels']).agg(
        gym_name = pd.NamedAgg('gym', lambda x: x.iloc[0]), #use the first gym name in the cluster as a representative, for readability
        N_wins = pd.NamedAgg('mma_w', 'sum'),
        N_total = pd.NamedAgg('mma_total', 'sum'))
```

Note: `NamedAgg()` returns `g` as a single-indexed dataframe, which will be easier to query later. It can be thought of as a drop-in replacement for SQL `AS`.
{: .notice--info}

To improve readability, I modify the `styler` object corresponding to `g` so that the win rate cell is colored in proportion to its value.

```python
s = g.sort_values(by='N_total', ascending=False).iloc[0:10,:].style
s.format(precision=2)
s.background_gradient(cmap='RdYlGn',vmin=0, vmax=1, subset=['win_rate'], axis=1)
```
Here is a look at the 10 largest gyms in the dataset:

<div>
<style scoped>
#T_8ec9b_row0_col3 {
  background-color: #fdbd6d;
  color: #000000;
}
#T_8ec9b_row1_col3 {
  background-color: #fbfdba;
  color: #000000;
}
#T_8ec9b_row2_col3 {
  background-color: #fff6b0;
  color: #000000;
}
#T_8ec9b_row3_col3 {
  background-color: #d1ec86;
  color: #000000;
}
#T_8ec9b_row4_col3 {
  background-color: #bfe47a;
  color: #000000;
}
#T_8ec9b_row5_col3 {
  background-color: #abdb6d;
  color: #000000;
}
#T_8ec9b_row6_col3 {
  background-color: #e8f59f;
  color: #000000;
}
#T_8ec9b_row7_col3, #T_8ec9b_row8_col3 {
  background-color: #d5ed88;
  color: #000000;
}
#T_8ec9b_row9_col3 {
  background-color: #c9e881;
  color: #000000;
}
</style>
<table>
  <thead>
    <tr>
      <th>gym_name</th>
      <th>N_wins</th>
      <th>N_total</th>
      <th>win_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Independent</td>
      <td>289</td>
      <td>880</td>
      <td id="T_8ec9b_row0_col3">0.33</td>
    </tr>
    <tr>
      <td>Team Quest</td>
      <td>181</td>
      <td>355</td>
      <td id="T_8ec9b_row1_col3">0.51</td>
    </tr>
    <tr>
      <td>DragonHouse MMA</td>
      <td>101</td>
      <td>215</td>
      <td id="T_8ec9b_row2_col3">0.47</td>
    </tr>
    <tr>
      <td>Millennia</td>
      <td>125</td>
      <td>203</td>
      <td id="T_8ec9b_row3_col3">0.62</td>
    </tr>
    <tr>
      <td>The Arena</td>
      <td>131</td>
      <td>202</td>
      <td id="T_8ec9b_row4_col3">0.65</td>
    </tr>
    <tr>
      <td>Team Alpha Male</td>
      <td>115</td>
      <td>167</td>
      <td id="T_8ec9b_row5_col3">0.69</td>
    </tr>
    <tr>
      <td>Bodyshop</td>
      <td>86</td>
      <td>153</td>
      <td id="T_8ec9b_row6_col3">0.56</td>
    </tr>
    <tr>
      <td>CSW</td>
      <td>88</td>
      <td>145</td>
      <td id="T_8ec9b_row7_col3">0.61</td>
    </tr>
    <tr>
      <td>CMMA</td>
      <td>86</td>
      <td>142</td>
      <td id="T_8ec9b_row8_col3">0.61</td>
    </tr>
    <tr>
      <td>Victory MMA</td>
      <td>82</td>
      <td>130</td>
      <td id="T_8ec9b_row9_col3">0.63</td>
    </tr>
  </tbody>
</table>
</div>

According to the data, the largest "gym" is actually the group of fighters with no gym affiliation. This group performs notably worse than the next largest gyms. This may imply that the benefits of belonging to a gym -- coaching, infrastructure, training partners -- have an effect on fighter outcomes. Alternatively, good fighters may be seeking out the best gyms and vice-versa. It is difficult to infer causality without individual fight and gym affiliation information as a function of time.

The tabular representation is fine for a few rows of information, but quickly becomes overwhelming as the data grow larger.

# Geospatial Visualization
A good visualization of large, multi-dimensional data is one that compresses multiple features into a relatively small number of plotted objects in an appealing way, without sacrificing information content. In this case, I use geography as the foundation to build a global visual representation of the data.

## Geopy
First, I had to *geocode* the data -- i.e., convert a list of gym names into map coordinates. [GeoPy](https://geopy.readthedocs.io/en/stable/) is a python library which allows a user to query the API of a variety of geocoding services. I used Google's geocoding API, which is not free in general, but allows a certain number of free queries per month.  

I query the API sequentially with every gym name in a given cluster until I get the coordinates for that gym, then go to the next cluster. This is a lazy way to to avoid excessively redundant calls to the API. The returned object is a `geopy.Point()` containing a latitude and longitude.

```python
#a null column to populate with coordinates
g['gym_lat'] = pd.NA
g['gym_lon'] = pd.NA

from geopy.geocoders import GoogleV3
geolocator = GoogleV3(api_key='<MY_API_KEY>')

from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=1, error_wait_seconds=10)

for gcl in g.index:
      gym_group = df[(df['gym_cluster_labels'] == gcl)]['gym']

      for gym_name in gym_group:
          location = geocode(query=gym_name, components=[('administrative_area', 'CA'), ('country', 'US')])

          #filter out null and default response from API
          if (location is not None) and (location.raw['place_id'] != 'ChIJPV4oX_65j4ARVW8IJ6IJUYs'):
              g.loc[gcl, 'gym_lat'] = location.latitude
              g.loc[gcl, 'gym_lon'] = location.longitude
              break
```

### Sidenote: Setting up the Geocoding API
On GCP, geocoding is handled by the [Geocoding API](https://developers.google.com/maps/documentation/geocoding/overview). To access the API, you have to first generate an API key. From the cloud console dashboard, enter the navigation menu, then go to **APIs & Services** &rarr; **Credentials**

<figure style="width: 300px" class="align-center">
  <img src="/assets/images/mma_analysis/gcp_geocoding_step1.png" alt="">
  <figcaption></figcaption>
</figure>

On the top of the **Credentials** page, click the **Create Credentials** button and select API key from the dropdown menu. The new key will show up on your credentials dashboard. In the row with the newly created key, select **Actions** &rarr; **Edit API Key**. Under **API restrictions**, select **Restrict Key**, then in the dropdown menu, check the **Geocoding API** and **OK**.

<figure style="width: 300px" class="align-center">
  <img src="/assets/images/mma_analysis/gcp_geocoding_step2.png" alt="">
  <figcaption></figcaption>
</figure>

Now if you click **SHOW KEY** on the credentials dashboard, you'll see the alphanumeric string you need to pass to geopy to authenticate the calls to the API.

Warning: The API key is linked to your billing account, so keep it protected!
{: .notice--danger}

## Folium
With the geocoded data, I generated an interactive map of the MMA gyms across California with [Folium](https://python-visualization.github.io/folium/), a python wrapper for leaflet.js. First, I generate the `folium.Map` object with a default map center and starting zoom value. Then, I define a RGBA colormap to color the map markers. Each marker is a circle centered on a gym location, with its size and color corresponding to the gym's total fight count and win rate, respectively. The unaffiliated fighters are represented by a marker placed off the coast of Santa Cruz:

```python
marker_data = g
marker_data.loc[413, 'gym_lon'] = -125. #shift the 'Independent' entry

import folium
ca_coordinates = (37.41928,-119.30639)
map = folium.Map(location=ca_coordinates,
                 zoom_start=6,
                 max_zoom=9,
                 tiles='cartodb dark_matter')

import branca.colormap as cm
#Red-White-Green colormap
gradient = cm.LinearColormap([(255,0,0,255), (255,255,255,125),(0,255,0,255)],
                              vmin=0., vmax=1.,
                              caption='win rate')

for idx,row in marker_data.iterrows():
    folium.Circle(
        location=[row.gym_lat, row.gym_lon],
        radius=row.N_total*100,
        weight=0, #stroke weight
        fill_color= gradient.rgba_hex_str(row.win_rate),
        fill_opacity=gradient.rgba_floats_tuple(row.win_rate)[-1]
    ).add_to(map)

map.add_child(gradient)

map
```

<figure class="align-center">
  <iframe src="/assets/html/mma_analysis/mma_map.html" style="height: 600px; width: 100%;" title=""></iframe>
  <figcaption>A map of MMA gyms in California. The size of each marker is scaled to the total number of fights of the gym, and the color corresponds to the percentage of fights won. The unaffiliated fighters are aggregated into a single marker shown off the coast of Santa Cruz.</figcaption>
</figure>

---
 <p style="font-size:0.75em">If you are curious about the thumbnail image for this project, see
<a href="https://gist.github.com/arana89/2071557bac0160f99bba3ece9f5e9ad0">here</a>
</p>
