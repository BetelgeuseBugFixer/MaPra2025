# MaPra2025 Monomer Structure Prediction
Here we present our "Master Praktikum" Project, where we attempt to predict 3D structure of proteins by predicting 
foldtoken and decoding them again

## How to run on lrz (only for cool people)
- Just navigate to our repro in the group folder on the lrz:
  ```bash
    cd /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/group1/repro/MaPra2025
  ```
- And tun the Project
  ```bash
    sbatch train.sbatch
  ```
- *IMPORTANT NOTE*: Do not edit the repro onn the lrz directly, instead edit it locally and the pull

## Update Dependencies
- This Project uses a modified Version of the [Pytorch Container from the NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- The modified container is stored in our group dir in `container/mapra_container.sqsh`
- To modify it, you need to start enter the container, modify it and export it again, as shown below
    ```bash
    # create new container from current sqsh file
    # this can be skipped if there is already a container with this file
    enroot create --name updated_container container/mapra_container.sqsh
  
    # now start container
    enroot start updated_container
  
    # inside the container you are free to make your changes, like
    pip install pyjokes
    
    #after you made your changes, exit the container
    exit
  
    #and finally overwrite the current sqsh file
    enroot export --output container/mapra_container.sqsh updated_container
    ```