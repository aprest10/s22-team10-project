# s22-team10-project
### Team Members: Andrew Preston, Matt Fadler, Riley Welch, Yassaman Nassiri, Jansen Long
---
<br>

# GEOGUESSR PROJECT
### Overview
---
We loaded a cartographic boundry shapefile of the U.S. available through the US Census Bureau ([Source](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html)) into mainlandGrid.ipynb. Next we isolated the mainland U.S. We overlayed it with a grid and combined the two to cut off the grids using the boundry of the U.S. A function called MergeFactor was used to combined sections that were too small after the previous process. This left us with a file called <ins>mainlandGrid.pickle</ins> that was a map of the mainland U.S. with 65 grids across it. Using this  Using a Convolution Neural Network 
