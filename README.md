# Brain-Inspired Capture: Evidence-Driven Neuromimetic Perceptual Simulation for Visual Decoding
![image](https://github.com/flysnow1024/BI-Cap/blob/main/Display/framework.png)
## Description
Visual decoding of neurophysiological signals is pivotal for Brain-Computer Interfaces (BCIs) and AI. Current approaches often fail to bridge the systematic gap and stochastic gap between neural and visual modalities, neglecting the intrinsic mechanisms of the Human Visual System (HVS). 
To bridge these gap, we propose Brain-Inspired Capture (BI-Cap), a \textbf{Neuromimetic Perceptual Simulation} paradigm that aligns modalities based on HVS processing. We implement four dynamic and static biologically plausible transformations, explicitly incorporating MI-guided dynamic blur regulation to simulate adaptive visual processing. Furthermore, addressing the inherent high dynamicity and inter-subject heterogeneity of neural activity, we introduce an \textbf{Evidence-Driven Latent Space Representation} framework. This approach facilitates robust neural representation by explicitly modeling the uncertainty within HVS processing. Extensive experiments on zero-shot brain-to-image retrieval tasks across two public benchmark datasets demonstrate the effectiveness of our paradigm, surpassing state-of-the-art methods by significant margins of 9.2/% and 8.0/%, respectively.
## Usage
### Setup
* **OS**: Linux
* **CUDA**: 11.8
* **Python**: 3.11.13
```bash
pip install -r requirements.txt
```
### Datasets
```text
./data
├── things_eeg
│   └── Preprocessed_data_250Hz_whiten
│       ├── sub-01
│       │   ├── train.npy
│       │   └── test.npy
│       ├── sub-02
│       │   ├── train.npy
│       │   └── test.npy
│       └── ...
├── things_meg
│   └── Preprocessed_data
│       ├── sub-01
│       │   ├── train.npy
│       │   └── test.npy
│       ├── sub-02
│       │   ├── train.npy
│       │   └── test.npy
│       └── ...
└── images_set (images dataset for train and test)
    └── test_images
    └── training_images 
