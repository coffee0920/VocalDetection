## DeepLearningAudioClassification

- **環境需求（Environment Requirement）**
    - Tensorflow : `2.3.0`
    - Python : `3.6.9`
    - Librosa : `0.8.0`
    - Tensorflow-probability : `0.11.0`
    - CUDA : `10.1.243`
- **環境設置（Setup）**
    1. First, download the dataset from [Github](https://github.com/NTUT-LabASPL/FMA-C-DataSet-for-Vocal-Detection)  and place it in the `data` folder.
    2. Depending on your requirements, execute different files:
        - **Tensorflow**
            - **Train/test SCNN18 model:**：
                - Run `pipelines.sh`, refer to `train.py`  for the meanings of different flags.
            - **Train/test SCNN18_SENet model:**
                - Run `pipelines.sh`, refer to `train.py`  for the meanings of different flags.
                - Add SE-Net to SCNN18 every two or three Convolution Layers
                - Consider to remove maxpooling since SE-Net will do maxpooling (see after Layer 17)
            - **Train/test SCNN18_SelfAttentionCNN model:**
                - Run `pipelines.sh`, refer to `train.py`  for the meanings of different flags.
                - Add Self-Attention Implementation in Convolutional Neural Network to SCNN18
                - Position to insert: After Layer 2, 4, 7, 9, 11, 13, 16, 18
                - Need a lot of Memory(RAM) during Training(62GB when All 8 Layers insert)
        - **Pytorch**
            - Only `SCNN18.ipynb` has the right loss calculation
            - Other models should calculate the average loss of each epoch(the original code accumulate the loss of every batch)
            - **Train/test SCNN18 model:**
                - Run `SCNN18.ipynb`
                - Learning rate should change back to 1.0 since 0.5 has not improve its performance
            - **Train/test SCNN18_MHSA model:**
                - Run `SCNN18_MHSA_1.ipynb` with one MHSA Layer insert after Layer 18 in SCNN18
                - Run `SCNN18_MHSA_1_mlp.ipynb` with one MHSA Layer insert after Layer 16 and add MLP After MHSA in SCNN18
                - Run `SCNN18_MHSA_8.ipynb` with eight MHSA Layers insert in SCNN18
                - Position to insert: After Layer 2, 4, 7, 9, 11, 13, 16, 18
            - **Train/test SW-MHSA model:**
                - Run `SW-MHSA.ipynb` 
                - The `SW-MHSA model` can only run on one GPU(Parallel Computing doesn't work)