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
