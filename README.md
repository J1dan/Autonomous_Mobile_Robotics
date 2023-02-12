# Autonomous_Mobile_Robotics

## Dependencies
* Python>=3.8
* ROS Melodic/Noetic
* open3d
* sklearn
* json

## Usage
1. Install the required libraries using pip, or 
* using conda:

   ```conda env create -f environment.yml```

2. Run the python or jupiter script in the folders

   `cd Perception`

   ```cd 1_lidar```

   ```python LiDAR_Clustering.py ```


## Demonstration

### LiDAR clustering

* DBSCAN

<img src="Perception/Examples/cls_ground.png" width="187"/><img src="Perception/Examples/ori_top.png" width="230"/>

* Meanshift&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;OPTICS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;K-means

<img src="Perception/Examples/meanshift.png" width="215"/><img src="Perception/Examples/optics.png" width="215"/><img src="Perception/Examples/kmeans.png" width="215"/>

* Agglomerative&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Birch

<img src="Perception/Examples/agglomerative.png" width="250"/><img src="Perception/Examples/birch.png" width="250"/>

<br>

### ROS implementation

<img src="Perception/Examples/output1.gif" width="400"/>
<img src="Perception/Examples/output2.gif" width="400"/>
