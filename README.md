# GraphLab NLP Project

This is a work in progress. In this repository we will be doing research on leveraging NLP applications in cybersecurity using the HDFS dataset.

## Setup Instructions

We have a script `fetch.py` which extracts the HDFS dataset over the internet. The HDFS dataset can be found [here](https://zenodo.org/record/3227177#.YopouOzML0o). For simplicity, we created a directory called `data` for storing all of our data files but our code is not dependent on file path. Fetching all of this data can take several minutes.

Sample run instructions:

> python fetch.py https://zenodo.org/record/3227177/files/HDFS_1.tar.gz data/HDFS_1.tar.gz data/HDFS_1

The dataset is very large and training can be very time-consuming. For simplicity, we created a script `filter.py` to filter out a smaller subset of HDFS blocks.

Sample run instructions:

> python filter.py data/HDFS_1/HDFS.log data/HDFS_1/anomaly_label.csv data/truncated_logs.txt 1000 True


### Contributors
<ul>

<li>Yejin Kim</li>
<li>Mark Bentivegna</li>

</ul>