# Assessing Visual Quality

## Quick Start
Step 1. Download the model weights and the dataset that you want to try.  
Step 2. Place the model weights and the dataset in the correct location..  
Step 3. Begin training and testing.  

### Some Tips
Before training and testing, please modify the batch size in *_train.py and *_test.py to fit your GPU memory.  

To train DOF:
```
python DOF_train.py
```

The default checkpoint will be saved in the checkpoint folder.

To perform validation, please modify DOF_test.py by setting type='validation' and updating the file read by df(df = pd.read_csv('dataset/DOF_dataset/DOF_val.csv')).  

```
python DOF_test.py
```

## From Scratch
To start from scratch, please complete the following steps:

1. Download DOF dataset(from huggingface: comHannah/bokeh-dataset).  
2. Do the preprocess(add blur, check the data, and so on).  
3. Use Q-Align as the pretrained model for mPLUG-Owl2, and record the points to the csv.  
4. Start training the model.  

## Model Weights and Dataset
(Uploading to google cloud)  
DOF model weights: https://drive.google.com/file/d/1ESRjAJsl0E38uW6FTQFgADqhiGDLRI7Q/view?usp=sharing  
BAID model weights: https://drive.google.com/file/d/1Zzfoy1Jm8bd-TKIoAFjrSYMxm4MLQaZH/view?usp=sharing  
AVA model weights: https://drive.google.com/file/d/1QDg71oh2q9JJUYO_Ose9cqQ6OhzTA6An/view?usp=sharing  

BISD: https://drive.google.com/file/d/1lxumX_yQdN8n1bA1BAtisDRPK67Dx5x2/view?usp=sharing  
BAID:  
AVA:  