# MOAA

## Overview
This repository contains Python implementation of the algorithm framework for Multi-Objective Adversarial Attack for FSE'24 review

## Code Structure
<pre>
MOAA/
├── README.md
├── requirement.txt
├── setup.py
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