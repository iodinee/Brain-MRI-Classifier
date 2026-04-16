# CNN-VGG16 Brain MRI Classification System

---

## Abstract

This project implements a convolutional neural network (CNN) for medical image classification using VGG16 transfer learning. It detects intracranial neoplasms from MRI scans and simulates a diagnostic inference pipeline.

The system includes an interactive interface and explainability via Grad-CAM heatmaps.

---

## Clinical Classification Targets

- Glioma: high-grade gliomas and astrocytomas  
- Meningioma: extra-axial, dural-based tumors  
- Pituitary adenoma: sellar/suprasellar lesions  
- No tumor: negative imaging findings  

---

## Methodology

### Architecture
- Backbone: VGG16 CNN  
- Approach: Transfer learning  
- Input: T1/T2-weighted MRI scans  
- Output: 4-class softmax classification  

### Explainability
Grad-CAM is used to highlight image regions influencing predictions.

---

## System Features

- MRI upload and inference interface  
- Neural activation visualization  
- Real-time predictions  
- Confidence scoring  
- Heatmap visualization (Grad-CAM)  

---

## Dataset

- ~5,700 MRI scans  
- Balanced multi-class dataset  
- Normalization + resizing  
- Data augmentation applied  

---

## Performance

- Accuracy: ~98%  
- Metrics: precision, recall, F1-score  
- Evaluation: held-out test set  

---

## Clinical Relevance

Demonstrates CNN-based decision support for neuroradiology classification tasks.

---

## Limitations

Not a clinical diagnostic tool.

---

## Future Additions

- DICOM pipeline integration  
- Multi-modal imaging (MRI + CT)  
- Clinical validation  
- Uncertainty estimation
