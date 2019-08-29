# Python data mining

The program implements hierarchical clustering using CURE algorithm. 
The algorithm works by clustering random samples of a large data set or database, eliminates outliers and then applies this
clustering over the entire data set. 
Program flow: 
1) First pass - cluster sample file, scan data in sample and assign to points closest to cluster
2) Find cluster representatives as those closest to centroid in a cluster C
3) Move cluster representatives by alpha 
4) Second pass - cluster entire data set 
 

Usage: cure.py k sample data-set n alpha

where k is the number of clusters, sample is the sample file, the data set contains the whole data set file,
n is the number of representatives in the cluster and alpha is the distance that the representatives are moved towards centroid. 


Packages used: 
- sys
- os 
- heapq
- numpy
- pandas
- itertools and combinations 
