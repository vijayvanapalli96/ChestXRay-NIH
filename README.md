# Deep Learning for Chest X-Ray Medical Diagnosis of Lung Cancer

## Problem Definition and Understanding

Lung cancer diagnosis from chest X-rays is critical for early treatment but remains challenging. This project leverages deep learning, specifically Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs), to automate the diagnostic process. Utilizing the NIH Clinical Center dataset of 100,000 annotated chest X-ray images, the goal is to build a scalable system that provides accurate, generalizable disease prediction, improving accessibility and efficiency in clinical settings.

The dataset includes 14 conditions:
- Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Mass, Nodule, Atelectasis, Pneumothorax, Pleural Thickening, Pneumonia, Fibrosis, Edema, and Consolidation.
  
This multi-label classification problem requires the model to recognize multiple conditions in a single X-ray, where conditions can co-exist.

## Data Preparation and Preprocessing
<img width="988" alt="image" src="https://github.com/user-attachments/assets/b0a714df-aab9-4ea1-b9cc-7d12f640544f">

- **Data Selection:** Used only images and labels for disease prediction, excluding irrelevant metadata (age, gender, view position).
- **Label Filtering:** Removed unreliable "No Finding" labels to focus on pathologies.
- **Image Path Management:** Full paths for easy access across multiple folders.
- **Label Encoding:** Encoded labels across features for multi-label classification.

### Data Preprocessing Steps
1. **Label Filtering**: Removed bad labels using an image label map.
2. **Oversampling**: Balanced minority classes by sampling with replacement.
3. **Image Transformations**:
   - Resize to 256x256, followed by a 224x224 crop.
   - Random horizontal flip (50% probability).
   - Normalization to standardize image pixel values.
4. **Class Weights**: Calculated class weights for loss function balancing, using BCEWithLogitsLoss with a sigmoid activation function.
   
<img width="942" alt="image" src="https://github.com/user-attachments/assets/96c5aed6-312c-4027-aedd-45b28cdc1b07">

## Model Selection and Development

Four models were evaluated for their multi-label classification effectiveness:
- **DenseNet-121**: Efficient in feature reuse, ideal for limited data with detailed features, reducing overfitting.
- **VGG16**: Simple architecture, though computationally demanding and parameter-heavy.
- **ResNet50**: Uses residual blocks to handle deep networks, balancing performance and computational efficiency.
- **Vision Transformer (ViT)**: Captures global dependencies via self-attention but struggled with this dataset due to overfitting.

## Generalization

- **Sampling Strategy**: Used random sampling to avoid bias in data representation, addressing the dataset's inherent imbalance.
- **Imbalance Handling**: Focused on reducing model bias towards the majority class, such as "No Finding," which was excluded. Minority classes like "Pneumonia" and "Hernia" were oversampled to prevent misclassification.

## Evaluation and Performance Metrics

<img width="447" alt="image" src="https://github.com/user-attachments/assets/ef416fc2-aaa4-46cc-b765-912d89f8b266">


- **DenseNet-121**: Best performer with balanced recall and precision, highest F1-score, and robust AUC and validation accuracy.
- **ResNet50**: Strong generalization with consistent validation AUC and accuracy.
- **VGG16**: Moderate performance, showing signs of overfitting.
- **ViT**: Poor adaptation to dataset, struggling with validation AUC and accuracy.

## Conclusion

The NIH Clinical Center chest X-ray dataset analysis with DenseNet-121, ResNet50, VGG16, and ViT models demonstrated the effectiveness of deep learning in automating lung cancer diagnosis. DenseNet-121 achieved the best balance between precision and recall, suggesting its suitability for clinical settings where accurate and reliable predictions are essential.

---

## Libraries Used
- **PyTorch**: For model development.
- **Torchvision**: For image transformations.
- **Pandas**: For data handling.
- **Scikit-learn**: For evaluation metrics and additional utilities. 

This project highlights the potential of advanced machine learning to enhance healthcare diagnostics, specifically in lung cancer detection from chest X-rays.
