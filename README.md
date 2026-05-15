# InksNet: studying historical manuscripts with AI

This repository contains code and datasets used in the study of iron-gall inks in historical manuscripts by InksNet neural network, laser ablation inductively coupled plasma mass spectrometry (LA-ICP-MS) and bathophenanthroline-soaked indicator papers.

## Background

### Iron-gall ink

<img align="right" alt="księga_rachunkowa_z_tłem" src="https://github.com/user-attachments/assets/6b3cccf2-a400-4cfb-8bf8-e4ee8a0e28e0" style="width:27%; height:auto;" />

Iron-gall ink is the most common and widespread agent used in handwriting in Europe for almost ten centuries. 

Due to the lack of universal preparation procedure, the ink composition varies greatly across manuscripts, regions and ages.

In favourable cases, the insight into the elemental composition of iron-gall inks can support formulation of new hypotheses about time, autorship and circumstances of manuscript creation process.

### LA-ICP-MS and indicator papers

Laser ablation inductively coupled plasma mass spectrometry (LA-ICP-MS) is a leading-edge analytical technique used for determination of elemental composition in materials. 
LA-ICP-MS is extremely sensitive, but unfortunately destructive - it requires sampling from the object examined.

To circumvent this limitation, for the strictly protected historical manuscripts we used bathophenanthroline-soaked indicator papers. 
With this method, one can transfer small amounts of ink onto the paper without damaging the object. After that, the indicator can be analysed in spectrometer. But how to retrieve the original information about ink elemental composition from the data on indicator only?

<img alt="la_icp_ms_and_indicators" src="https://github.com/user-attachments/assets/436bb930-8fba-49aa-ac78-dfe679aa2d50" style="width:75%; height:auto;" />


## InksNet

<img align="right" alt="nn_v5" src="https://github.com/user-attachments/assets/930d5ff0-34ff-437e-9d2c-deb2fdc193f7" style="width:35%; height:auto;" />

InksNet is a feed-forward neural network predicting the elemental composition of an ink based on the LA-ICP-MS measurements of bathophenanthroline-soaked papers. 

The input and output are vectors of length 11, corresponding to 11 crucial elements present in iron-gall inks (Al, S, Mn, Co, Cu, Zn, Pb, Fe, Mg, Na, K).

InksNet has 6 hidden linear layers with sizes : 93, 586, 660, 743, 105, 893, each followed by activation function GeLU and dropout with probability of approx. 0.08. 

The architecture of InksNet was chosen using Optuna search.

The network was trained to minimize L1 loss function by AdamW optimizer with learinig rate of approx. $9 \cdot 10^{-4}$ and weight decay of approx. $3 \cdot 10^{-8}$ for 3000 epochs.

## Datasets
### Nicolaus Copernicus handwritten notes

The data come from marginalia authored by Mikołaj Kopernik (Nicolaus Copernicus) in incunabula and early printed books from the historic collection of the Hosianum Higher Theological Seminary Library in Olsztyn (Poland). On the basis of palaeographic and contextual analysis, the volumes were attributed as having belonged to or been used by Mikołaj Kopernik. The studied corpus includes copies of Sentences by Peter Lombard (shelf marks Inc. 61), the homiletic-theological treatise Quaestiones evangeliorum by Juan de Torquemada (shelf mark Inc. 70), composite bindings of legal by Baldus de Ubaldis and Jacobus Alvarottus (shelf mark Inc. 202) and two Postils by Albertus Magnus (shelf mark Sd. 367).

### Constitution of 3rd May 1791

<img align="right" alt="Konstytucja_księga_z_tłem" src="https://github.com/user-attachments/assets/c6fe3fb7-5b81-40f6-99a3-92bdc77d33e2" style="width:35%; height:auto;"/>

The Constitution of 3 May 1791 is a government act written for the Polish-Lithuanian Commonwealth, adopted after two years of deliberations by the Great (Four-Year) Sejm on 3rd May 1791. Its uniqueness lies in being the first constitution in Europe (and the second one in the world, after the Constitution of the United States) to be enacted in a democratic manner. Today, three handwritten original copies are preserved at the Central Archives of Historical Records in Warsaw, each belonging to one of the three different archival volumes: Lithuanian Metrica (Metryka Litewska, ML), Potocki Public Archive (Archiwum Publiczne Potockich, APP) and Archive of the Four-Year Sejm (Archiwum Sejmu Czteroletniego, ASC). Apart from the main text, each copy includes some corrections and deputies’ signatures. Additionally, manuscript from ASC has some editorial corrections and marginal annotations, and in contrast to the other two archives it bears only the Sejm’s marshals’ signatures – of Stanisław Małachowski and Kazimierz Sapieha. 

### *Merkuriusz polski* encyclopedia

<img align="right" alt="image" src="https://github.com/user-attachments/assets/b8896a6f-eb5d-4325-bc6c-e2bc7743a9df" style="width:15%; height:auto;" />


*Merkuriusz polski* is a seventeenth-century verse encyclopedia of practical knowledge for landowners written by Jakub Kazimierz Haur, Polish author best known for his practical handbooks on managing landed property and households. The manuscript of the book, stored in the Manuscript Department of the University of Warsaw Library, is characteristic for its unique, complex form. Most of the folios, in addition to the main text, were supplemented with numerous marginal notes, parallel text in an additional column, and countless smaller and larger fragments of paper pasted onto the pages forming the main body of the manuscript. The added inserts contain supplements and newer versions of the text. The author was refining and correcting the text, it is also possible that he decided to change the content of some of the encyclopedic entries and wanting a clean copy, he chose to paste in a new version rather than crossing out the old text. It seems clear that the manuscript presents the text in the process of its creation and is likely not the author’s final version. However, the final form of the book is not known — it is unclear whether it was never completed, destroyed, or lost. 

### Artificial inks

This additional dataset contains contemporary, hand-made inks with atypical composition.

### Training dataset

The original fragments of archival manuscripts written with iron-gall inks were investigated for the purpose of constructing dataset used for training the neural network. Manuscript fragments were obtained from the Central Archives of Historical Records in Warsaw, Reference Material Collection of the Heritage Science Laboratory in Ljubljana, and private collections. They belonged to archives with various origins and dating, therefore allowing to reflect the potential diversity of ink composition and preservation state.

## Repository

The aforementioned datasets are located in `data/` folder. To fully reproduce our analysis, run scripts from `analysis/` folder in the order indicated by file names. Source code is available in `src/` folder, while the trained model can be found in `models/`. The prediction for consecutive datasets is located in `results/`. Finally, `figures/` folder contains visualisations summarising the training process and the results obtained.

## Web application

LA-ICP-MS data from indicator papers require specialised, multi-step preprocessing that extracts valuable information. To facilitate this process, we created a web application called tINKer. To use it, go [HERE](https://bioputer.mimuw.edu.pl/tinker/main). tINKer enables uploading your own data, performing LA-ICP-MS-specific preprocessing, running InksNet on the obtained input, and downloading the prediction.

## Acknowledgements

This work was supported by the Polish National Science Centre grants: Spectral Analysis of Legacy Inks using machiNe leArning: ALiNA (2021/41/B/ST4/02860) and Optimal-transport based algorithms for Mass Spectrometry and NMR (2021/41/B/ST6/03526). Funding for access to the Reference Material Collections of the Heritage Science Laboratory at the University of Ljubljana was provided through I0-E012 (Slovenian Research and Innovation Agency).

## Citing

Paper in preparation. Meanwhile, if you are using code or data from this repository, please cite it using url.
