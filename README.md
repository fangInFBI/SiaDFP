# SiaDFP
This is the code for the "SiaDFP: A Disk Failure Prediction Framework Based on Siamese Neural Network in Large-Scale Data Center" and baselines.

The baseline of traditional mechina learning methods are implement based on the https://github.com/SidiLu001/disk_failure_prediction 

Pre-request

> CentOS 7.9.2009
> python 3.7.6
> PyTorch 1.2.0
> CUDA Version: 10.2

Usage

- Clone the repo: https://github.com/fangInFBI/SiaDFP.git.
- Download the dataset from [Dataset](https://www.dropbox.com/scl/fo/ehqrmzkx9ndz5gfmfommx/h?dl=0&rlkey=fasg953wzlq01gwk0gjnimekk) and put the data into the folder dataset/dataset1 or dataset/dataset2.
- Refer to the codes of corresponding sections for specific purposes.


Model
- loadData.py
> The code to load the disk dataset.

- SiaDFPNet.py
> The codes of the models in this paper.

Train&&Test
- trainTest.py
> The codes to train SiaDFP and evaluate the model on the test data.
