# MMTL-UniAD: A Unified Framework for Multimodal and Multi-Task Learning in Assistive Driving Perception

<div align="center">
  <img src="framework.jpg" alt="TLogo" width="600"/>
</div>

## Introduction



Advanced driver assistance systems require a comprehensive
understanding of the driver’s mental/physical state and
traffic context but existing works often neglect the potential
benefits of joint learning between these tasks. This paper
proposes MMTL-UniAD, a unified multi-modal multitask
learning framework that simultaneously recognizes
driver behavior (e.g., looking around, talking), driver emotion
(e.g., anxiety, happiness), vehicle behavior (e.g., parking,
turning), and traffic context (e.g., traffic jam, traffic
smooth).

## Environment Setup

### System Requirements

- Python 3.7+
- CUDA 11.0+ (for GPU acceleration)
- 24GB+ RAM

###  Installation Steps

1.  **Clone the repository**

```bash
git clone https://github.com/Wenzhuo-Liu/MMTL-UniAD.git
```

2.  **Create a virtual environment**

```bash

conda create -n MMTL python=3.8
conda activate MMTL

```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```
## Dataset

The project is trained, validated, and tested using the AIDE dataset. For dataset-related files or more information, please search for "AIDE Dataset" online or visit its official repository.


## Data Preprocessing

```bash
python Crop.py
```

## Usage

### Running the Model

Use the `run.py` script to train or test the model. The script supports two modes: training mode and testing mode.

```python
# Modify the mode parameter in run.py
mode = "train"  # Training mode
# mode = "test"  #  Testing mode

# Then run the script
python run.py
```

