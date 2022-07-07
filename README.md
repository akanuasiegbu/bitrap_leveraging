# BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal Estimation
Yu Yao, Ella Atkins, Matthew Johnson-Roberson, Ram Vasudevan and Xiaoxiao Du

This repo contains the code for our paper:[BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal Estimation](https://arxiv.org/abs/2007.14558).

Our BiTraP-NP network architecture:

<img src="figures/bitrap_np.png" width="800">

Our BiTraP-GMM decoder architecture:

<img src="figures/bitrap_gmm.png" width="600">

## Installation
### Dependencies
Our code was implemented using python and pytorch and tested on a desktop computer with Intel Xeon 2.10GHz CPU, NVIDIA TITAN X GPU and 128 GB memory.

* NVIDIA driver >= 418
* Python >= 3.6
* pytorch == 1.4.1 with GPU support (CUDA 10.1 & cuDNN 7)

Run following command to add bitrap path to the PYTHONPATH

  cd bidireaction-trajectory-prediction
  export PYTHONPATH=$PWD:PYTHONPATH

One can also use docker with `docker/Dockerfile`. 

## Step 1: Download Dataset
 * The extracted bounding box trajectories for Avenue and ShanghaiTech with the anomaly labels appended can be found [here](https://drive.google.com/drive/folders/1MNpbhB9LS7k0X_fK8BZWqGRoVfPxANZl?usp=sharing) .
 * To want to recreate the input bounding box trajectory 
   * Download [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) and [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html) dataset 
   * Use [Deep-SORT-YOLOv4](https://github.com/LeonLok/Deep-SORT-YOLOv4/tree/a4b7d2e1263e6f1af63381a24436c5db5a4b6e91) commit number a4b7d2e
  
 ## Step 2: Training
 ### Training BiTrap Model
*  For training BiTrap models refer forked repo [here](https://github.com/akanuasiegbu/bidireaction-trajectory-prediction).

Train on Avenue Dataset
```
python tools/train.py --config_file configs/avenue.yml
```

Train on ShanghaiTech Dataset
```
python  tools/train.py --config_file configs/st.yml
```

To train/inferece on CPU or GPU, simply add `DEVICE='cpu'` or  `DEVICE='cuda'`. By default we use GPU for both training and inferencing.

Note that you must set the input and output lengths to be the same in YML file used (```INPUT_LEN``` and ```PRED_LEN```) and ```bitrap/datasets/config_for_my_data.py``` (```input_seq``` and ```pred_seq```)

 
 
 
 ## Step 3: Inference 
##### Pretrained BiTrap Model:
Trained BiTrap models for Avenue and ShanghiTech can be found [here](https://drive.google.com/drive/folders/1942GF9FIzoqTVOHyW2Qo86s3R1OOSnsg?usp=sharing) 

##### BiTrap Inference
To obtain BiTrap PKL files containing the pedestrain trajectory use commands below.
Test on Avenue dataset:
```
python tools/test.py --config_file configs/avenue.yml CKPT_DIR **DIR_TO_CKPT**
```

Test on ShanghaiTech dataset:
```
python tools/test.py --config_file configs/st.yml CKPT_DIR **DIR_TO_CKPT**
```


##### PKL Files
 BiTrap pkl files can be found [here](https://drive.google.com/drive/folders/1ELYuty5kg-J14jrDH66Gv9rhn58O1t9I).
 
 * Download the ```output_bitrap``` folder which contains the pkl file folders for Avenue and ShanghiTech dataset.
 * Naming convention: ```in_3_out_3_K_1``` means input trajectory and output trajectory is set to 3. And K=1 means using Bitrap as unimodal.
 


## Citation

If you found the repo is useful, please feel free to cite:
```
@article{yao2020bitrap,
  title={BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal Estimation},
  author={Yao, Yu and Atkins, Ella and Johnson-Roberson, Matthew and Vasudevan, Ram and Du, Xiaoxiao},
  journal={arXiv preprint arXiv:2007.14558},
  year={2020}
}
```
