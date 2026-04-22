# InksNet: study of iron-gall inks in historical manuscripts using machine learning and LA-ICP-MS

This repository contains code and datasets used in the study of iron-gall inks in historical manuscripts by InksNet neural network. 

## Background

### Iron-gall ink

<img width="426" height="145" alt="image" src="https://github.com/user-attachments/assets/7c2965d6-a8ac-4889-b824-baa9805a7efe" />

Iron-gall ink is the most common and widespread agent used in handwrting in Europe for almost ten centuries. 

Due to the lack of universal preparation procedure, the ink composition varies greatly across manuscripts, regions and ages.

In favourable cases, the insight into the elemental composition of iron-gall inks can support formulation of new hypotheses about time, autorship and circumstances of manuscript creation process.

### LA-ICP-MS and Bphen indicator papers

Laser ablation inductively coupled plasma mass spectrometry (LA-ICP-MS) is a leading-edge analytical technique used for determination of elemental composition in materials. 
LA-ICP-MS is extremely sensitive, but unfortunately destructive - it requires sampling from the object to be examined.

To circumvent this limitation, for the strictly protected historical manuscripts we used bathophenanthroline-soaked indicator papers. 
With this method, one can transfer trace amounts of ink onto the paper without damaging the object. After that, the indicator can be analysed in spectrometer.

## InksNet

<img width="1984" height="1376" alt="nn_v5" src="https://github.com/user-attachments/assets/930d5ff0-34ff-437e-9d2c-deb2fdc193f7" />


InksNet is a feed-forward neural network predicting the elemental composition of an ink based on the LA-ICP-MS measurements of bathophenanthroline-soaked papers. 

The input and output are vectors of length 11, corresponding to 11 crucial elements present in iron-gall inks (Al, S, Mn, Co, Cu, Zn, Pb, Fe, Mg, Na, K).

InksNet has 6 hidden linear layers with sizes : 93, 586, 660, 743, 105, 893, each followed by activation function GeLU and dropout with probability of approx. 0.08. 

The architecture of InksNet was chosen using Optuna search.

The network was trained to minimize L1 loss function by AdamW optimizer with learinig rate of approx. $9 \cdot 10^{-4}$ and weight decay of approx. $3 \cdot 10^{-8}$ for 3000 epochs.

## Datasets
### Nicolaus Copernicus handwritten notes

### Constitution of 3rd May 1791

<img width="2016" height="1512" alt="Konstytucja_księga_z_tłem" src="https://github.com/user-attachments/assets/c6fe3fb7-5b81-40f6-99a3-92bdc77d33e2" />

### *Merkuriusz polski* encyclopedia

### Artificial inks

### Training dataset

<img width="2040" height="1536" alt="księga_rachunkowa_z_tłem" src="https://github.com/user-attachments/assets/6b3cccf2-a400-4cfb-8bf8-e4ee8a0e28e0" />

### Repository



### Web application

## Citing

If you are using code or data from this repository, please cite ...
