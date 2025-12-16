# E. coli and K. pneumoniae in Urine Cultures Using YOLOv12s on Chromogenic Agar Plate Images**

---

## Overview

This repository provides the source code, trained model weights, and representative sample images for the study:

> *(Original Research)*

The proposed framework employs a **YOLOv12s-based deep learning model** to automatically detect and classify *E. coli* and *K. pneumoniae* colonies directly from **chromogenic agar urine culture images**, enabling rapid pathogen identification and supporting antimicrobial stewardship.

---

## Motivation

Urinary tract infections (UTIs) are among the most prevalent bacterial infections worldwide, with *E. coli* and *K. pneumoniae* being the dominant pathogens. Conventional urine culture workflows require **16–72 hours**, often delaying targeted therapy and increasing the risk of inappropriate broad-spectrum antibiotic use.

This study introduces an **AI-assisted approach** that enables:
- Rapid colony-level detection and classification
- Earlier pathogen identification (≈16 hours)
- Reduced diagnostic turnaround time
- Improved antibiotic stewardship and risk mitigation, particularly for vulnerable patient groups

---

## Dataset

- **Total images:** 1,547 chromogenic agar Petri dish images  
  - *E. coli*: 850 images  
  - *K. pneumoniae*: 697 images
- **Source:** Copan Walk Away Specimen Processor (WASP) system  
- **Collection period:** November 2023 – February 2024
- **Imaging:** 16 MP tri-linear color camera under calibrated LED lighting

### Dataset Split
- **TrainDB:** 100 images (50 *E. coli*, 50 *K. pneumoniae*)  
- **TestDB:** 1,447 images (801 *E. coli*, 646 *K. pneumoniae*)
- **External OOD Test Set:** 91 images from an independent published dataset

Each Petri dish image contains **dozens of bacterial colonies**, enabling effective object-level learning despite a limited number of training images.

---

## Proposed AI Model

- **Architecture:** YOLOv12s (You Only Look Once, v12 – small)
- **Training strategy:** Transfer learning using Ultralytics pretrained weights
- **Mode:** Object detection (colony-level)
- **Input size:** 640 × 640 pixels
- **Classes:**
  - ID1 → *Escherichia coli*
  - ID2 → *Klebsiella pneumoniae*

Detected colonies are aggregated using a **majority voting mechanism** to produce **image-level classification**.

---

## Data Augmentation

To enhance generalization, the following augmentations were applied during training:

- Rotations (±90°, 180°, ±15° random)
- Gaussian blur
- Translation (±10%)
- Scaling (±10%)
- Shear (up to 10°)
- Horizontal (50%) and vertical (20%) flips
- Color jitter:
  - Hue ±1.5%
  - Saturation ±70%
  - Value ±40%

No augmentation was applied during testing or external validation.

---

## Performance

### Internal Test Set (TestDB)
| Metric | Value |
|------|------|
| Accuracy | 0.99 |
| Precision | 0.99 |
| Recall | 0.99 |
| F1-score | 0.99 |
| Specificity | 0.99 |
| FPR | 0.001 |
| FNR | 0.004 |
| Hamming Loss | 0.002 |

### External OOD Test Set
- **Accuracy:** 1.00  
- **Precision / Recall / F1:** 1.00  

> ⚠️ Note: External validation relied on color-based assumptions rather than MALDI-TOF confirmation.

---


## 👥 Authors & Affiliations

- **Abdurrahman Sarmis**  
  Department of Microbiology Laboratory,  
  Goztepe Prof. Dr. Suleyman Yalcin City Hospital, Istanbul, Türkiye  

- **Serpil Ustebay**  
  Department of Computer Engineering,  
  Istanbul University–Cerrahpaşa, Istanbul, Türkiye  

- **Muhammed A. Mutlu**  
  Department of Microbiology Laboratory,  
  Fatih Sultan Mehmet Training and Research Hospital, Istanbul, Türkiye  

- **Furkan Adem Canbaz**  
  Department of Pediatric Surgery,  
  Sancaktepe Sehit Prof. Dr. Ilhan Varank Training and Research Hospital, Istanbul, Türkiye  

- **Gulsum Kubra Kaya**  
  Faculty of Engineering and Applied Sciences,  
  Cranfield University, Bedford, UK  

---

## Ethics & Disclosure

- Ethical approval ID: **2024/188**
- Patient confidentiality preserved according to institutional policies
- **No conflicts of interest declared**

---

## Correspondence Author
**Abdurrahman Sarmis, MD**  
Microbiologist  
Email: asarmis@gmail.com  
ORCID: 0000-0002-8156-6633

## Code Maintainer:

**Serpil Ustebay, PhD**  
Computer Engineer – AI Model Development & Code Implementation  
Department of Computer Engineering  
Istanbul University–Cerrahpaşa, Istanbul, Türkiye  
Email: serpil.ustebay@istanbul.edu.tr  



