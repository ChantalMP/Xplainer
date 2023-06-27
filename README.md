# Xplainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis

This is the official repository for the paper "Xplainer: From X-Ray Observations to Explainable Zero-Shot Diagnosis" (https://arxiv.org/pdf/2303.13391.pdf), which was accepted for publication at MICCAI 2023. 

We propose a new way of explainability for zero-shot diagnosis prediction in the clinical domain. Instead of directly predicting a diagnosis, we prompt the model to classify the existence of descriptive observations, which a radiologist would look for on an X-Ray scan, and use the descriptor probabilities to estimate the likelihood of a diagnosis, making our model explainable by design. For this we leverage BioVil, a pretrained CLIP model for X-rays and apply contrastive observation-based prompting. We evaluate Xplainer on two chest X-ray
datasets, CheXpert and ChestX-ray14, and demonstrate its effectiveness
in improving the performance and explainability of zero-shot diagnosis.

Demo: https://huggingface.co/spaces/Chantal/Xplainer

### Installation:
1. Clone this repository
   ```
   git clone https://github.com/ChantalMP/Xplainer
   ```
2. Install requirements:
   
   - use Python 3.7
   - install requirements:
   ```
   pip install hi-ml-multimodal==0.1.2
   pip install -r requirements.txt
   ```
   
3. Download data

### Reproduce our results:
run
```
python -m inference --dataset chexpert
```
or
```
python -m inference --dataset chestxray14
```

### Run demo locally:
run
```
python -m demo
```
