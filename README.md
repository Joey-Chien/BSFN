# Assessing Visual Quality

## Quick Start

Step 1. Download the model weights and the dataset that you want to try.  
Step 2. Place the model weights and the dataset in the correct location.  
Step 3. Begin training and testing.  

### Requirements
* Python >= 3.9
- Pytorch >= 1.12.0  
* Torchvision >= 0.13.0  

```
conda create --name bsfn
source activate bsfn
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pandas
pip install scipy
pip install tqdm
pip install opencv-python
pip install transformers
pip install torchmetrics
pip install torchmetrics[multimodal]
```



### Some Tips
1. Before training and testing, please modify the batch size in *_train.py and *_test.py to fit your GPU memory.  

2. Put the images to the right place and modify the ```images_dir``` in *_train.py and *_test.py.

3. To test and val the DOF:
```
python DOF_test.py
```
To perform validation, please modify DOF_test.py by setting ```type='validation'``` and updating the file read by df ```df = pd.read_csv('dataset/DOF_dataset/DOF_val.csv')```  

4. To train the DOF:
```
python DOF_train.py
```
The default checkpoint will be saved in the checkpoint folder.




## From Scratch
To start from scratch, please complete the following steps:

1. Download DOF dataset (from huggingface: comHannah/bokeh-dataset).  
2. Do the preprocess (e.g., add blur, check the data integrity, etc.).  
3. Use Q-Align as the pretrained model for mPLUG-Owl2, and record the points in the csv.  
4. Begin training the model.  

## Model Weights and Dataset
(Uploading to google cloud)  
DOF model weights: https://drive.google.com/file/d/1ESRjAJsl0E38uW6FTQFgADqhiGDLRI7Q/view?usp=sharing  
BAID model weights: https://drive.google.com/file/d/1Zzfoy1Jm8bd-TKIoAFjrSYMxm4MLQaZH/view?usp=sharing  
AVA model weights: https://drive.google.com/file/d/1QDg71oh2q9JJUYO_Ose9cqQ6OhzTA6An/view?usp=sharing  

BISD: https://drive.google.com/file/d/1lxumX_yQdN8n1bA1BAtisDRPK67Dx5x2/view?usp=sharing  
BAID: https://drive.google.com/file/d/1QSsgk96_owOGB_qTRcDMFo7Ix1bf1jQZ/view?usp=sharing  
AVA: https://drive.google.com/file/d/1DI1hFM5Vjk-bPUYAe0mR79CUZd1BKppH/view?usp=sharing  