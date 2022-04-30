<h1 align="center">Demo</h1>

Temp location of link to google drive: https://drive.google.com/drive/folders/1RZKKJQbPvG1PoUpcjkIwt54VZd9Z83LG

## Grid Set-up

---
```python
pip install shapely
```

```python
pip install geopandas
```

```python
pip install gmaps
```

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

```python
states = gpd.read_file('/path/to/cb_2018_us_nation_5m.shp')
```

```python
states
```

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

```python
plt.plot(mainland[:,1],mainland[:,0])
plt.show()
```

```python
pickle.dump(mainland,open("/path/to/mainland.pickle","wb"))
mainland = pickle.load(open("/path/to/mainland.pickle", "rb"))
mainland = Polygon(np.flip(mainland))
x,y = mainland.exterior.xy
plt.plot(x,y)
```
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

```python
mainlandGrid = partition(mainland, base, mergeFactor=0.2)
len(mainlandGrid)
```

```python
for i in mainlandGrid.values():
    plt.plot(i[:,1],i[:,0])
plt.show()
```

```python
pickle.dump(mainlandGrid,open("/to/path/mainlandGrid.pickle","wb"))
```

## Data Scrapping

---
### Training data

```python
# Insert API key
key = ''
import requests
import json, os
import urllib.request
import random
dataDir = "/path/to/training_data"
```

```python
# Data is scraped from all grids
# searchGrids = mainlandGrid.keys()

# Data is scraped for first 3 grids
searchGrids = list(range(0,3))
print("Search in Grids: {}".format("All" if searchGrids==mainlandGrid.keys() else searchGrids))
```

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

```python
# Change to save images to testing data folder
dataDir = "/path/to/testing_data"
```

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

```python
plt.imshow(x_train[6])
```

```python
# Load testing data images
filelist = glob.glob('/path/to/testing_data/*.jpg')
```

```python
x_test = np.array([np.array(Image.open(fname)) for fname in sorted(filelist)])
print(x_test.shape)
```

```python
plt.imshow(x_test[0])
```

## Create y_train and y_test

```python
y_train = []
for i in range(1560):
    y_train.append(i // 24)
y_train = np.array(y_train)
```

```python
print(y_train)
```

```python
y_test = []
for j in range(390):
    y_test.append(j // 6)
y_test = np.array(y_test)
```

```python
print(y_test)
```

## Create Network Architecture

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

```python
keras.utils.plot_model(model,show_shapes=True,expand_nested=True)
```
