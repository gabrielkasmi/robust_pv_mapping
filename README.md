# Space-scale Exploration of the Poor Reliability of Deep Learning Models: the Case of the Remote Sensing of Rooftop Photovoltaic Systems

## Set-up

First, clone the repositories of Kymatio and the WCAM into the folder. 

```
pip install kymatio

cd robust_pv_mapping
git clone https://github.com/gabrielkasmi/spectral-attribution.git
```

Install the required packages:

``` 
pip install -r requirements.txt
```

Finally, download the model weights on this Zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12179554.svg)](https://zenodo.org/records/12179554)
 and make sure that you've downloaded the training dataset BDAPPV, accessible [here](https://zenodo.org/records/12179554).

## Overview

The folder is organized in notebooks, located in the `notebook` folder. Each notebook enables you to replicate a part of the analysis. The notebook `quantitative-eval` displays the results of the evaluation of different models trained on Google images and evaluated on IGN images. The notebook `quantitative-eval` displays the result of the analysis of the sensitivity of CNNs to distorsions in the lower scales. The notebook `scattering-transform` focuses on the deployment of the Scattering transform on Google and IGN images. 

The weights of the CNN models are accessible on the Zenodo repository (folder `models`). Run the script ``scattering-train.py` to train Scattering tansform-based classifiers on BDAPPV. Alternatively, you can load the weights from the Zenodo repository (folder `weights-scattering`). The weight for the Scattering transform are organized in `google` and `ign` folders, for models trained on Google and IGN respectively. The depth of the models ($m=1$ to $m=3$) are indicated in the model names.  

## Citation 

```
@inproceedings{kasmi2023can,
  title={Can We Reliably Improve the Robustness to Image Acquisition of Remote Sensing of PV Systems?},
  author={Kasmi, Gabriel and Dubus, Laurent and Saint-Drenan, Yves-Marie and Blanc, Philippe},
  booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
  url={https://www.climatechange.ai/papers/neurips2023/10},
  year={2023}
}
```


