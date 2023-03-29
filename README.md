# Autonomous_Mobile_Robotics

## Dependencies
* Python>=3.6
* ROS Melodic/Noetic
* Open3d
* sklearn
* json

## Usage
1. Install the required libraries using pip, or 
* using conda:

   ```terminal
   conda env create -f environment.yml
   ```

2. Run the python or jupiter script in the folders

* Clustering
   ```bash
   cd Perception
   ```

   ```bash
   cd 1_lidar
   ```

   ```terminal
   python LiDAR_Clustering.py
   ```

* ICP
   ```terminal
   cd ICP
   ```

    ***Argument 1***: --task: *task1*, *task2*

    ***Argument 2***: --method: for task1, *none*, *downSampling*; for task2, *none*, *downSampling*, *globalReg*, *combined*


   ```terminal
   python ICP.py --task=task1 --method=downSampling
   ```

* Planning
   ```terminal
   python Planning/src/plan.py 
   ```

   *You can select different planners in plan.py*

## Demonstration

### LiDAR clustering

* DBSCAN

<img src="Perception/Examples/cls_ground.png" width="187"/><img src="Perception/Examples/ori_top.png" width="230"/>

* Meanshift&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;OPTICS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;K-means

<img src="Perception/Examples/meanshift.png" width="215"/><img src="Perception/Examples/optics.png" width="215"/><img src="Perception/Examples/kmeans.png" width="215"/>

* Agglomerative&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Birch

<img src="Perception/Examples/agglomerative.png" width="250"/><img src="Perception/Examples/birch.png" width="250"/>

<br>

### Clustering ROS implementation

<img src="Perception/Examples/output1.gif" width="400"/>

<br>

### ICP
<img src="ICP/Example/combined2.png" width="350"/> <-- Applying ICP with down-sampling and global registration for 60 iterations

### A* algorithm and its variations
<img src="Planning/Examples/a_h_exp.png" width="350"/> <-- Node expansion of hybrid-A*

<img src="Planning/Examples/bidirectional.png" width="350"/> <-- Paths planned by bidirectional-A*
<img src="Planning/Examples/bf.png" width="350"/> <-- Optimal path for traveling seller problem solved by brute-force method
