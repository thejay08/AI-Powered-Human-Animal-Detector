# Human and Animal Detection System

## Overview
This project implements an AI-based system to detect and classify humans and animals in images and videos. The system uses OpenAI's CLIP model for feature extraction and classification, adhering to the requirement of not using pre-trained models like YOLO or Faster R-CNN. The system is capable of:

- Detecting humans and animals in images or video feeds.
- Classifying animals into specific categories (e.g., cow, goat, lion, etc.).
- Triggering alerts when a human or animal is detected.

The system achieves these objectives while maintaining modularity and scalability for future improvements.

## Features
- **Human and Animal Detection:** Identifies humans and animals in images or videos.
- **Animal Classification:** Classifies animals into predefined categories such as "wild," "pet," or "farm."
- **Alert System:** Prints alerts when humans or animals are detected.
- **Batch Processing:** Optimized for GPU acceleration with batch processing for faster inference.
- **Video Support:** Processes video files with frame sampling and parallel processing for efficiency.
- **Evaluation:** Includes a validation module to evaluate the model's accuracy on a test dataset.

## Approach

### 1. Model Selection
The system uses OpenAI's CLIP model (ViT-B/32) for feature extraction and classification. CLIP is a vision-language model that maps images and text into a shared embedding space, enabling zero-shot classification.

### 2. Region Proposal
To detect objects in an image, the system generates region proposals using:

- **Sliding Window:** Generates regions of varying scales and aspect ratios.
- **Grid-based Regions:** Divides the image into grids for better coverage.
- **Selective Search:** Uses OpenCV's selective search for additional region proposals.

### 3. Preprocessing
Each region is resized to 224x224 pixels and normalized using a custom preprocessing pipeline. This ensures compatibility with the CLIP model and efficient memory usage.

### 4. Classification
The system tokenizes text descriptions of predefined categories (e.g., "a photo of a lion") and computes their embeddings using CLIP. For each region, the image embedding is compared with text embeddings to classify the object.

### 5. Non-Maximum Suppression (NMS)
To handle overlapping detections, the system applies NMS to retain the most confident detection for each object.

### 6. Parallel Processing
For video processing, frames are processed in parallel using Python's ThreadPoolExecutor. This significantly reduces processing time for large video files.

### 7. Alert System
Alerts are triggered when a human or animal is detected. For example:

- "⚠️ ALERT: Wild animal detected!"
- "👤 ALERT: Person detected!"

### 8. Evaluation
The system includes a validation module to evaluate accuracy on a test dataset. Metrics such as precision, recall, and F1-score are calculated to assess performance.

## Challenges Faced

### Region Proposal Efficiency
Generating region proposals for large images was computationally expensive. To address this, a combination of sliding windows, grid-based regions, and selective search was used.

### Memory Management
Processing large batches of regions on the GPU caused memory overflow. This was mitigated by dynamically adjusting batch sizes based on available GPU memory.

### Model Limitations
CLIP is not specifically trained for object detection, so its performance depends on the quality of region proposals. This required extensive tuning of region generation parameters.

### Video Processing
Processing high-resolution videos in real-time was challenging. Frame sampling and parallel processing were implemented to optimize performance.

### Accuracy Requirement
Achieving 80% accuracy required careful selection of categories and thresholds. Extensive testing and validation were performed to meet this requirement.

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/prashantsingh5/animal_and_person_detection.git
   cd animal_and_person_detection
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

Ensure you have a compatible GPU and CUDA installed for optimal performance.

## Usage

### 1. Image Processing
To process an image:
```
python parallel_detection.py
# Replace the image_path variable in the main() function with the path to your test image.
```

### 2. Video Processing
To process a video:

```
python parallel_detection.py
# Replace the video_path variable in the main() function with the path to your test video.
```

### 3. Process Entire Dataset
```bash
python test_model.py --dataset "path/to/dataset" --output "path/to/output"
```

### 4. Validation
To evaluate the model on a test dataset:
```
python validation.py
# Replace <dataset_path> with the path to your test dataset.
```

### 5. Testing the model
```
python test_model.py --dataset "Enter the dataset path"
```


## File Structure
- `parallel_detection.py`: Main implementation of the detection system with parallel processing.
- `Validation.py`: Module for evaluating the model on a test dataset.
- `detection2.py`: Alternative implementation with GPU-specific optimizations.
- `single.py`: Simplified version for single-image processing.
- `requirements.txt`: List of required Python packages.

## Results
The system achieves over 80% accuracy on the test dataset, meeting the project requirements. Alerts are triggered for detected humans and animals, and the processed images/videos are saved with bounding boxes and labels.