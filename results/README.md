# Results

Four deep learning architectures were evaluated for multi-class classification of thyroid scintigraphy images.

## Model Performance

| Model          | Accuracy | 95% Confidence Interval |
| -------------- | -------: | ----------------------: |
| DenseNet-201   |  **91%** |             88.76–93.11 |
| ConvNeXt Tiny  |  **92%** |             89.94–94.05 |
| ConvNeXt Small |  **90%** |             87.92–92.43 |
| ResNet-50      |  **89%** |             85.43–90.38 |

**ConvNeXt Tiny achieved the highest overall classification accuracy of 92% (95% CI: 89.94–94.05).**

ConvNeXt Tiny and DenseNet-201 demonstrated strong class-wise sensitivity and specificity across the five diagnostic categories.

ROC-AUC analysis indicated that the disease-specific uptake patterns provided good discriminative power for classification.

### Temporal Validation

Temporal validation using previously unseen thyroid scintigraphy images demonstrated good model generalizability to more recent clinical data.

### Key Findings

* **Best overall accuracy:** ConvNeXt Tiny — **92%**
* **DenseNet-201 accuracy:** **91%**
* **ConvNeXt Small accuracy:** **90%**
* **ResNet-50 accuracy:** **89%**
* ConvNeXt Tiny and DenseNet-201 showed strong class-wise diagnostic performance.
* ROC-AUC analysis demonstrated good discriminatory ability.
* Temporal validation supported the generalizability of the developed models.
