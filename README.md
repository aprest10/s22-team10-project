# s22-team10-project
### Team Members: Andrew Preston, Matt Fadler, Riley Welch, Yassaman Nassiri, Jansen Long
<br>

# GEOGUESSR PROJECT
### Overview
---
##### Grid Making
We loaded a cartographic boundry shapefile of the U.S. available through the US Census Bureau ([Source](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)) into mainlandGrid.ipynb. Next we isolated the mainland U.S. We overlayed it with a grid and combined the two to cut off the grids using the boundry of the U.S. A function called MergeFactor was used to combined sections that were too small after the previous process. This left us with a file called <ins>mainlandGrid.pickle</ins> that was a map of the mainland U.S. with 65 grids across it. Each section of the grid was associated with a number 0-64. 

##### Data Scrapping
Using this map data and a data scrapper we scrapped images from Google maps using the Google maps API. For each grid section we scrapped 10 locations (650 location total). For each location we got 3 pictures, since Google map images are 360-degree images we used the API to grab an image of size 200px by 400px at 0, 90, and 180-degrees. 






Using a Convolution Neural Network 
