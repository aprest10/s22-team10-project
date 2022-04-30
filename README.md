# s22-team10-project
<h1 align='center'> GEOGUESSR PROJECT</h1>
<p align='center'> <strong>Team Members:</strong><br> Andrew Preston<br> Matt Fadler<br> Riley Welch<br> Yassaman Nassiri<br> Jansen Long<p>
<br>

### Overview
---
##### Grid Making
We loaded a cartographic boundary shapefile of the U.S. available through the US Census Bureau ([Source](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)) into <ins>mainlandGrid.ipynb</ins>. <p align= 'center'>![Unedited Map](/images/us_all.png)</p> Next we isolated the mainland U.S. We overlayed it with a grid and combined the two to cut off the grids using the boundry of the U.S. A function called MergeFactor was used to combined sections that were too small after the previous process. This left us with a file called <ins>mainlandGrid.pickle</ins> that was a map of the mainland U.S. with 65 grids across it. Each section of the grid was associated with a number 0-64. 

##### Data Scrapping
Using this map data and <ins>datascrapper.ipynb</ins> we scrapped images from Google maps using the Google maps API. For each grid section we scrapped 10 locations (650 location total). For each location we got 3 pictures (1950 pictures total), since Google map images are 360-degree images we used the API to grab an image of size 200px by 400px at 0, 90, and 180-degrees. For Training/Testing we used an 80%/20% split; 8 locations or 24 images per section for training and 2 locations or 6 images per section for testing. The format for labeling each image is [grid no.]_[unique no.].jpg (ex. 0_5.jpg).

##### Model Training




Using a Convolution Neural Network 
