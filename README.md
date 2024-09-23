# MOAA

## Overview
This repository contains Python implementation of the algorithm framework for Multi-Objective Adversarial Attack for FSE'24

## Code Structure
<pre>
MOAA/
├── README.md
├── requirement.txt
├── parser/
├── defect_detection/
│   ├── dataset/
│   ├── saved_models/
│   │   ├── CodeBERT/
│   │   │   └── pytorch_model.bin
│   │   ├── assisted_model/
│   │   │   └── pytorch_model.bin
│   │   └── ...
│   ├── attack.py
│   ├── moaa.py
│   ├── model.py
│   ├── finetune.py
│   └── README.md
├── clone_detection/
│   └── ...
└── ...
</pre>



## Requirements
    - Python version: tested in Python 3.9.18
    - Pythorch: tested in 2.2.0
    - Transformers: tested in 4.37.2
    - Pandas: tested in 2.2.0
    - tree-sitter: tested in 0.20.4

## Assisted Model
the assisted model can be download from this [link](https://www.dropbox.com/scl/fo/lnpsmhz8r2t0v5zf54icm/ADCNILAttESRwpmLh2U8xVY?rlkey=9w78dvgehz9tpndmj369yvpt4&st=pa0d0c7x&dl=0).
