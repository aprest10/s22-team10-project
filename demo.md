<h1 align="center">Demo</h1>

Temp location of link to google drive: https://drive.google.com/drive/folders/1RZKKJQbPvG1PoUpcjkIwt54VZd9Z83LG?usp=sharing

## Grid Set-up

---
Before starting the following three commands should be run to install shapely, geopandas, and gmaps. Shapely and geopandas allows for easier manipulation of our data; while gmaps allows for code integrations with the Google maps API.
```python
pip install shapely
```

```python
pip install geopandas
```

```python
pip install gmaps
```
The following imports will be needed to create the grid and scrape the data from Google maps.
```python
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon
import numpy as np
import pickle
import gmaps
```
Any shapefile can be used depending on the scope of your project. This project is using the nation shapefile from the [U.S. Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html). Note: When working with shapefiles you will have multiple other files besides the actual shapefile. Please keep these files together, the shapefile will <ins>NOT</ins> work without the other files.
```python
states = gpd.read_file('/path/to/cb_2018_us_nation_5m.shp')
```

```python
states
```
![output](/images/2.png)<br><br>
The following code will output the graphical representation of the shapefile, with the center of the mainland U.S. marked.
```python
%%capture --no-display
mainland_center = Point(-98.35,39.50)
for i,j in enumerate(states['geometry'][0]):
    x,y = j.exterior.xy
    plt.plot(x,y)
    if mainland_center.within(j):
        mainland = np.array([(yj,xj) for xj,yj in zip(x,y)])
plt.scatter([-98.35], [39.50], c='r', marker='x', s=200)
plt.xlim([-190,-60])
plt.ylim([15,75])
plt.show()
```
![output](/images/3.png)<br><br>
Now the mainland can be isolated.
```python
plt.plot(mainland[:,1],mainland[:,0])
plt.show()
```
![output](/images/4.png)<br><br>
Pickle will be used to save this now map of the mainland. Pickle allows for saving of data and loading that same data without the risk of losing progress. The code below shows that the map is saved as [mainland.pickle](geoguessr/data/pickled_data/mainland.pickle) and that data can then be loaded back unchanged whenever needed.
```python
pickle.dump(mainland,open("/path/to/mainland.pickle","wb"))
mainland = pickle.load(open("/path/to/mainland.pickle", "rb"))
mainland = Polygon(np.flip(mainland))
x,y = mainland.exterior.xy
plt.plot(x,y)
```
![output](/images/4.png)<br><br>
Next a grid is overlayed on the map. The number of grids can be adjusted by changing the value of 'base'. Increasing 'base' will decrease the number of grids, while decreasing 'base' will increase the number of grids.
```python
value = mainland.bounds
base = 4

min_x = int(value[0]//base)
min_y = int(value[1]//base)
max_x = int(value[2]//base)
max_y = int(value[3]//base)

for i in range(min_x, max_x + 1):
for j in range(min_y, max_y + 1):
y = shapely.geometry.box(i*base, j*base, (i+1)*base, (j+1)*base)
r = mainland.intersection(y)
x,y = y.exterior.xy
plt.plot(x,y,c='y')
if r.is_empty:
continue
elif type(r) == shapely.geometry.multipolygon.MultiPolygon:
for gems in r.geoms:
x,y = gems.exterior.xy
plt.plot(x,y,c='g')
else:
x,y = r.exterior.xy
plt.plot(x,y,c='g')
plt.show()
```
![output](/images/5.png)<br><br>
The following chunk of code will take the map and grid overlay and perform two functions. One function performed will be the combining of the grid and map to prevent and overflow of the grid. The other function will merge sections that are too small with adjacent grids. The magnatude of effect by mergeFactor can be controlled by the value passed to mergeFactor. A larger mergeFactor will result in larger grid sections; while a smaller mergeFactor will result in smaller sections.
```python
def partition(mainland, base, mergeFactor):
    '''
    polygon: Unsplit polygon of mainland US
    dim: The dimensions of each grid to split the map into
    mergeFactor: threshold of smallest grid. 
    Any grid smaller will be combined with neighbouring grids
    '''
    value = mainland.bounds
    min_x = int(value[0]//base)
    min_y = int(value[1]//base)
    max_x = int(value[2]//base)
    max_y = int(value[3]//base)
    grid = 0
    res = []
    for i in range(min_x, max_x+1):
        for j in range(min_y, max_y+1):
            y = shapely.geometry.box(i*base, j*base, (i+1)*base, (j+1)*base)
            r = mainland.intersection(y)
            if r.is_empty:
                continue
            if type(r)==shapely.geometry.multipolygon.MultiPolygon:
                for gems in r.geoms:
                    res.append(gems)
                    grid += 1
            else:
                res.append(r)
                grid += 1
    return merge(res, mergeFactor)

def merge(polyList, mergeFactor):
    '''
    polyList: list of polygon grids the map is split into
    mergeFactor: threshold of smallest grid. 
    Any grid smaller will be combined with neighbouring grids
    '''
    def combine(pidx, polyL):
        p = polyL[pidx]
        del polyL[pidx]
        for idx,i in enumerate(polyL):
            u = p.union(i)
            if p.intersects(i) and type(u)!=shapely.geometry.multipolygon.MultiPolygon:
                polyL[idx] = u
                break
        return polyL
    
    mnLimit = max(polyList, key=lambda x:x.area).area * mergeFactor
    mnPoly = min(polyList, key=lambda x:x.area)
    while(mnPoly.area<=mnLimit):
        polyList = combine(polyList.index(mnPoly), polyList)
        mnPoly = min(polyList, key=lambda x:x.area)
        
    result = {}
    for idx,i in enumerate(polyList):
        x,y = i.exterior.xy
        result[idx] = np.array([(y,x) for x,y in zip(x,y)])
    return result

def plotMap(mainlandGrid):
    gPoly = []
    gMarkLoc = []
    gMarkInf = []
    info_box_template = """
    <dl>
    <dd>{}</dd>
    </dl>
    """
    for k,v in mainlandGrid.items():
        gPoly.append(gmaps.Polygon(
                        list(v),
                        stroke_color='red',
                        fill_color='blue'
                        ))
        gMarkLoc.append((v[0][0],v[0][1]))
        gMarkInf.append(info_box_template.format(k))
    fig = gmaps.figure(center=(39.50,-98.35), zoom_level=4, map_type='TERRAIN')
    fig.add_layer(gmaps.drawing_layer(features=gPoly))
    fig.add_layer(gmaps.marker_layer(gMarkLoc, info_box_content=gMarkInf))
    return fig
```
Call the above code and print the number of sections in the final map. This particualar example will produce 65 sections indexed 0-64.
```python
mainlandGrid = partition(mainland, base, mergeFactor=0.2)
len(mainlandGrid)
```
65
```python
for i in mainlandGrid.values():
    plt.plot(i[:,1],i[:,0])
plt.show()
```
![output](/images/6.png)<br><br>
The final grid can be seen above. This map is now saved into a pickle file for use later.
```python
pickle.dump(mainlandGrid,open("/to/path/mainlandGrid.pickle","wb"))
```

## Data Scraping

---
### Set-up

To get started with the data scraping, a Google Cloud account is required. We have provided a link below to get you started.

[Google Cloud Console](https://console.cloud.google.com)

The next step is to set up a Google Street View API set up on your Google Cloud Console. The link below provides details on how you are able to scrape images using your API key. Tutorials can be found on the Google Cloud Console to get your account billing set up, but a free $300 credit was provided by Google at the time of this project.

[Google Street View API Overview](https://developers.google.com/maps/documentation/streetview/overview)

Now that we have an API key, we can get started with scraping. The key has been left out below, but the variable is still set up for you to paste your new key into. A file directory to store both the training and testing images should be created, and the dataDir variable should store this file path.

```python
# Insert API key
key = ''
import requests
import json, os
import urllib.request
import random
dataDir = "/path/to/training_data"
```
The code below is utilizing the mainlandGrid.keys to create the list of grids to search for images in. An example is shown to scrape for images in both all grids and for only the first three grids.

```python
# Data is scraped from all grids
# searchGrids = mainlandGrid.keys()

# Data is scraped for first 3 grids
searchGrids = list(range(0,3))
print("Search in Grids: {}".format("All" if searchGrids==mainlandGrid.keys() else searchGrids))
```
![output](/images/7.png)<br><br>
We have specified that each image scraped will be 400x200 pixels. The location will be randomly generated using poly.bounds as the range for these numbers. The count variable allows you to change the number of locations to search for within each grid section. The trial variable allows you to attempt scraping an image from (or near) that coordinate a specificed amount of times before moving on. The ignum variable specifies the image number for that grid. For our project, we used 10 locations per grid section (8 for training 2 for testing) 3 images per location, so this means we pulled 24 images per grid section for training and 6 images for testing.

### Training data
For training data ignum iterates from 0 to 23 for the training locations becuase we are scraping images with a heading of 0, 90, and 180 degrees. This number is attached to the end of the image file name to ensure you scrape the same number of images per location. If the scraping does not work for a specified grid, you can add a conditional in the while loop to look like grid==gridMissing and it will only scrape images for that one grid location. You will also need to update the while loop to reflect the number of images you are missing  (i.e. count<3 would mean you need 3 more image locations).

```python
base = 'https://maps.googleapis.com/maps/api/streetview'
ext = '?size=400x200&location={}&fov=100&heading={}&radius={}&pitch=10&key={}'
print("Seacrchin Grids: {}".format("All" if searchGrids==mainlandGrid.keys() else searchGrids))
for grid,coor in mainlandGrid.items():        
    poly = Polygon(np.flip(coor))
    minx, miny, maxx, maxy = poly.bounds
    count = 0
    trials = 0
    locList = []
    if grid in searchGrids:
        saveFolder = dataDir
        if os.path.exists(saveFolder)==False:
            os.mkdir(saveFolder)
        locList = os.listdir(saveFolder)
        print("################## Searching grid {} ###################".format(grid))
        imgnum = 0
        
        # Only scraping 8 locations for training data (3 pictues from each location)
        while count<8 and trials<4:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            location = str(pnt.y)+','+str(pnt.x)
            if (poly.contains(pnt)) and (location not in locList):
                metaUrl = base + '/metadata' + ext.format(location, 0, 10000, key)
                r = requests.get(metaUrl).json()
                trials += 1
                print("Trial: {}, count: {}".format(trials,count))
                if r['status']=='OK' and poly.contains(Point(r['location']['lng'],r['location']['lat'])):
                    location = str(r['location']['lat'])+','+str(r['location']['lng'])
                    if (location not in locList):
                        print("Valid location found: {}".format(location))
                        locList.append(location)
                        saveFile = saveFolder
                        if os.path.exists(saveFile)==False:
                            os.mkdir(saveFile)

                        for heading in [0,90,180]:
                            imgUrl = base + ext.format(location, heading, 10000, key)
                            urllib.request.urlretrieve(imgUrl,saveFile+'/{}_{}.jpg'.format(grid,imgnum))
                            imgnum += 1
                        count += 1
                        trials = 0
                    else:
                        print("Failed trial {} location exists".format(trials))
                        print("Location {}".format(location))
                else:
                    print("Failed trial {} status or contains".format(trials))
                    print("Location {}".format(location))
        print("No duplicates: {}".format(len(locList)==len(set(locList))))
```
### Testing data

The same strategy used above is used to generate images for the testing data. The dataDir variable should be changed to the path for your testing data folder.

```python
# Change to save images to testing data folder
dataDir = "/path/to/testing_data"
```
The count variable is limited to 2, so only two locations for images will be used in the testing data. This will be a total of 6 images, 20% of the total of images per grid in the training data set.

```python
base = 'https://maps.googleapis.com/maps/api/streetview'
ext = '?size=400x200&location={}&fov=100&heading={}&radius={}&pitch=10&key={}'
print("Seacrchin Grids: {}".format("All" if searchGrids==mainlandGrid.keys() else searchGrids))
for grid,coor in mainlandGrid.items():        
    poly = Polygon(np.flip(coor))
    minx, miny, maxx, maxy = poly.bounds
    count = 0
    trials = 0
    locList = []
    if grid in searchGrids:
        saveFolder = dataDir
        if os.path.exists(saveFolder)==False:
            os.mkdir(saveFolder)
        locList = os.listdir(saveFolder)
        print("################## Searching grid {} ###################".format(grid))
        imgnum = 0
        
        # Only scraping 2 locations for testing data (3 pictues from each location)
        while count<2 and trials<4:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            location = str(pnt.y)+','+str(pnt.x)
            if (poly.contains(pnt)) and (location not in locList):
                metaUrl = base + '/metadata' + ext.format(location, 0, 10000, key)
                r = requests.get(metaUrl).json()
                trials += 1
                print("Trial: {}, count: {}".format(trials,count))
                if r['status']=='OK' and poly.contains(Point(r['location']['lng'],r['location']['lat'])):
                    location = str(r['location']['lat'])+','+str(r['location']['lng'])
                    if (location not in locList):
                        print("Valid location found: {}".format(location))
                        locList.append(location)
                        saveFile = saveFolder
                        if os.path.exists(saveFile)==False:
                            os.mkdir(saveFile)

                        for heading in [0,90,180]:
                            imgUrl = base + ext.format(location, heading, 10000, key)
                            urllib.request.urlretrieve(imgUrl,saveFile+'/{}_{}.jpg'.format(grid,imgnum))
                            imgnum += 1
                        count += 1
                        trials = 0
                    else:
                        print("Failed trial {} location exists".format(trials))
                        print("Location {}".format(location))
                else:
                    print("Failed trial {} status or contains".format(trials))
                    print("Location {}".format(location))
        print("No duplicates: {}".format(len(locList)==len(set(locList))))
```

## Convert to numpy array

Currently two folders containing the training images and testing images are available. Each image name is formated (grid no.) _ (0-23).jpg for training and (grid no.) _ (0-5).jpg for testing.<br><br> These pictures need to be converted into numpy arrays to be used by the model.<br> The structure for the numpy arrays are modeled after the tuple of numpy arrays found in OLA8.

Creating your numpy arrays is encuraged but premade arrays can be found on ~~[Google Drive](https://drive.google.com/drive/folders/1RZKKJQbPvG1PoUpcjkIwt54VZd9Z83LG?usp=sharing)~~ while available. (Issues occur with linked files - Making your own is recommended)

* x_train - holds all of the training images.
    * Expected shape (# of images, width, height, 3)
* y_train - holds the corresponding grid number for each element in x_train.
    * Expected shape (# of images)
* x_test - holds all of the testing images.
    * Expected shape (# of images, width, height, 3)
* y_test - holds the corresponding grid number for each element in x_test.
    * Expected shape (# of images)

### Create x_train and x_test
```python
from PIL import Image
import glob

# Load training data images
filelist = glob.glob('/path/to/training_data/*.jpg')
```

```python
x_train = np.array([np.array(Image.open(fname)) for fname in filelist])
print(x_train.shape)
```
![output](/images/8.png)
```python
plt.imshow(x_train[6])
```
![output](/images/9.png)
```python
# Load testing data images
filelist = glob.glob('/path/to/testing_data/*.jpg')
```

```python
x_test = np.array([np.array(Image.open(fname)) for fname in sorted(filelist)])
print(x_test.shape)
```
![output](/images/10.png)
```python
plt.imshow(x_test[0])
```
![output](/images/11.png)
### Create y_train and y_test

```python
y_train = []
for i in range(1560):
    y_train.append(i // 24)
y_train = np.array(y_train)
```

```python
print(y_train)
```
![output](/images/12.png)
```python
y_test = []
for j in range(390):
    y_test.append(j // 6)
y_test = np.array(y_test)
```

```python
print(y_test)
```
![output](/images/13.png)
## Create Network Architecture

For this project a Convolution Neural Network was used with keras.Sequential(). The use of Sequential allowed for the model to train sucessfully without running into memory issues. Implementing checkpoints was considered but not used, adding checkpoints would be recommended if attempting to train a more robust version the model.

```python
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from IPython.display import display
from mpl_toolkits.mplot3d import axes3d
%matplotlib inline
from tensorflow.keras import layers
```

```python
model = keras.Sequential()
model.add(layers.Input(x_train.shape[1:]))
model.add(layers.Conv2D(8, kernel_size=(16,16), activation='relu'))
model.add(layers.Conv2D(16, kernel_size=(32,32), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(16,16)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
# Output Logits (64)
model.add(layers.Dense(len(np.unique(y_train))))
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
optimizer=keras.optimizers.Adam(),
metrics=[keras.metrics.SparseCategoricalAccuracy()])

model.summary()
```
![output](/images/14.png)
```python
keras.utils.plot_model(model,show_shapes=True,expand_nested=True)
```
![output](/images/15.png)
## Train model
[placeholder]
## Load model

```python
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
```

```python
model = load_model('/path/to/model.h5')
```


## Run test
[placeholder]
## Output
[placeholder]



![NUmbered Grid](images/numberedGrid.png)
