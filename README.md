**PatchUNet: Lightweight Nuclei Segmentation in Histopathology Images**  
This repository contains the code and trained model for nuclei segmentation in histopathology images using a lightweight and resource-efficient U-Net architecture. The pipeline leverages a patch-based training strategy and data augmentation to enable effective segmentation even in constrained computational environments.

**Overview**  
Model: Lightweight U-Net (PatchUNet)  
Domain: Nuclei Segmentation in Histopathology Images  

**Key Features:**  
Patch-wise training for memory efficiency  

Combined Binary Cross-Entropy + Dice Loss  

EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks  

Augmented dataset generation from limited training samples  

**Dataset**  
Original Training Patches: 370  

Augmented Patches: 370  

Total Patches: 740  

Train/Validation Split: 592 / 148 (≈ 80/20)  

*Dataset used:* 
MoNuSeg 2018  

**Model Architecture**  
The model follows a U-Net encoder-decoder structure:  

Encoder: Convolution + MaxPooling  

Decoder: Transposed Convolutions  

Skip Connections: Preserve spatial features  

Optimized for patch-wise input with reduced memory footprint      
**Input and Pre-processing**  

<img width="312" height="176" alt="image" src="https://github.com/user-attachments/assets/54c1aae0-0be7-4c8a-8aec-7d89c3e0d800" />      

  
**Patch Extraction and Input**  

<img width="184" height="136" alt="patch extraction" src="https://github.com/user-attachments/assets/e5479243-8272-4157-bf35-652c358b7d0b" />  
<img width="602" height="166" alt="image" src="https://github.com/user-attachments/assets/9d509ab5-4c67-4440-9ff3-fa05c71bd7f0" />  
 
**Encoder Module**  

<img width="1194" height="769" alt="Model1 - Encoder" src="https://github.com/user-attachments/assets/e518f0cf-e729-4473-9593-44520b45a569" />  
**BottleNeck Layer**  
<img width="1191" height="470" alt="Model1 - Bottleneck" src="https://github.com/user-attachments/assets/de84b59c-bcad-46b8-94a3-30179998f4ab" />  
**Decoder Module**  
<img width="1192" height="755" alt="Model1 - Decoder" src="https://github.com/user-attachments/assets/e9e06b4e-f4d2-4305-89c7-4582bbd9c3ad" />  

**Output**  
<img width="123" height="143" alt="image" src="https://github.com/user-attachments/assets/9c5d4bb6-f0f1-4e5b-a1c1-8daa39026910" />  


**Training Configuration**  
*Parameter	Value*  
Epochs	50  
Batch Size	4  
Optimizer	Adam 
Loss Function	Binary Cross-Entropy + Dice Loss  
Callbacks	EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  

**Results**  
The model performs well on nuclei segmentation, retaining fine-grained spatial details while being computationally efficient. Sample outputs and evaluation metrics will be added soon.

**Getting Started**
  
**1. Clone the Repository**  
git clone (https://github.com/musmansharif/SegNetModel.git)  
cd patchunet-nuclei-segmentation  

**2. Install Dependencies**  
pip install -r requirements.txt  

**3. Train the Model**  
python train.py  

**Contributions**   

This project introduces:  
A lightweight U-Net variant tailored for histopathological nuclei segmentation  

A patch-based training pipeline that supports high-resolution images with limited resources  

An efficient augmentation and splitting strategy to boost performance

  Contact
For any queries or collaborations, please contact:  
📧 m.usman.sharif1995@gmail.com  
🏫 Lecturer, Riphah International University, Islamabad  

🔗 GitHub: @musmansharif
