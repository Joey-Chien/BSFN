# Assessing Visual Quality

## Training and Testing
Before running `DOF_train.py`, please modify the batch size in `DOF_train.py` to fit your GPU memory.

```
python DOF_train.py
```

The default checkpoint will be saved in the checkpoint folder.

To perform validation, please modify DOF_test.py by setting type='validation' and updating the file read by df.

```
python DOF_test.py
```

## To Re-produce:
For the reproduction, please complete the following steps:

1. Download BISD (Bokeh image scoring dataset).
2. Use Q-Align as the pretrained model for mPLUG-Owl2, and record the points.
3. Start training the model.
