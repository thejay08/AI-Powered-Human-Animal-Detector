import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

class AnimalHumanDetector:
    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Define categories
        self.categories = {
            'person': 'person',
            'bird': 'wild',
            'cat': 'pet',
            'dog': 'pet',
            'horse': 'farm',
            'sheep': 'farm',
            'cow': 'farm',
            'elephant': 'wild',
            'bear': 'wild',
            'zebra': 'wild',
            'giraffe': 'wild',
            'tiger': 'wild',
            'lion': 'wild',
            'deer': 'wild',
            'monkey': 'wild',
            'rabbit': 'pet',
            'hamster': 'pet',
            'goat': 'farm',
            'pig': 'farm'
        }
        
        # Prepare text features for CLIP
        self.text_inputs = torch.cat([
            clip.tokenize(f"a photo of a {category}") 
            for category in self.categories.keys()
        ]).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        # Colors for visualization
        self.color_map = {
            'pet': (0, 255, 0),     # Green
            'wild': (0, 0, 255),    # Red
            'farm': (255, 165, 0),  # Orange
            'person': (255, 0, 255) # Purple
        }
        
        # Detection parameters
        self.confidence_threshold = 0.5  # Lower threshold for whole image processing

    def detect_whole_image(self, image):
        """Process the entire image at once"""
        original_image = image.copy()
        height, width = image.shape[:2]
        
        # Convert to PIL image for CLIP
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Preprocess for CLIP
        try:
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return [], original_image
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        
        # Get top prediction
        values, indices = similarity[0].topk(3)  # Get top 3 predictions
        
        detections = []
        for i in range(len(values)):
            confidence = values[i].item()
            class_idx = indices[i].item()
            
            class_name = list(self.categories.keys())[class_idx]
            animal_type = self.categories[class_name]
            
            # Apply confidence threshold
            if confidence >= self.confidence_threshold:
                # Use the whole image dimensions for the bounding box
                # Add small margins to not cover the entire image
                margin_x = width * 0.05
                margin_y = height * 0.05
                
                detections.append({
                    'class': class_name,
                    'type': animal_type,
                    'confidence': confidence,
                    'bbox': [int(margin_x), int(margin_y), 
                             int(width - margin_x), int(height - margin_y)]
                })
                
                # Only take the highest confidence detection
                break
        
        return detections, original_image

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image"""
        for detection in detections:
            class_name = detection['class']
            animal_type = detection['type']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # Get color based on animal type
            color = self.color_map.get(animal_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{class_name} ({confidence:.2f})"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

    def process_image(self, image):
        """Process image and return detections"""
        start_time = time.time()
        
        # Detect objects using whole image approach
        detections, original_image = self.detect_whole_image(image)
        
        # Draw detections
        processed_image = self.draw_detections(original_image.copy(), detections)
        
        # Determine dominant type
        dominant_type = defaultdict(float)
        for detection in detections:
            dominant_type[detection['type']] += detection['confidence']
        
        detected_type = max(dominant_type.items(), key=lambda x: x[1])[0] if dominant_type else "unknown"
        
        # Trigger alert if needed
        if detected_type == 'wild':
            print("⚠️ ALERT: Wild animal detected!")
        elif detected_type == 'person':
            print("👤 ALERT: Person detected!")
        
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} seconds")
        
        return detections, detected_type, processed_image

    def process_video(self, video_path, output_path=None):
        """Process video file and save output"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        # Create output video writer if path is provided
        if output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        detections_summary = defaultdict(float)
        frame_count = 0
        alert_frame_count = 0
        last_detections = []  # Store previous detections
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 5th frame to speed up processing
            if frame_count % 5 == 0:
                detections, detected_type, processed_frame = self.process_image(frame)
                last_detections = detections  # Save detections for use in intermediate frames
                
                for detection in detections:
                    # Weight by confidence
                    detections_summary[detection['type']] += detection['confidence']
                
                # Trigger alert if needed
                if detected_type == 'wild':
                    alert_frame_count += 1
                    if alert_frame_count % 30 == 0:  # Alert every ~1 second (assuming 30fps)
                        print(f"⚠️ ALERT: Wild animal detected at frame {frame_count}!")
                
                if output_path:
                    out.write(processed_frame)
            else:
                # Use previous detections for intermediate frames
                if last_detections:
                    processed_frame = self.draw_detections(frame, last_detections)
                    if output_path:
                        out.write(processed_frame)
                elif output_path:
                    # No detections yet, just write the original frame
                    out.write(frame)
        
        cap.release()
        if output_path:
            out.release()
        
        # Determine the dominant category based on confidence-weighted detections
        detected_type = max(detections_summary.items(), key=lambda x: x[1])[0] if detections_summary else "unknown"
        
        return dict(detections_summary), detected_type

# The evaluate_model and main functions remain unchanged
def evaluate_model(detector, test_data_path):
    """Evaluate model accuracy on test dataset"""
    total = 0
    correct = 0
    
    # Test data format: directory with subdirectories for each class
    for class_dir in os.listdir(test_data_path):
        class_path = os.path.join(test_data_path, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        # Get ground truth class and type
        if class_dir.lower() in detector.categories:
            true_class = class_dir.lower()
            true_type = detector.categories[true_class]
        else:
            continue
            
        # Process each image in the class directory
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(class_path, img_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
                
            total += 1
            
            # Detect objects
            detections, detected_type, _ = detector.process_image(image)
            
            # Check if the correct class was detected
            detected_classes = [d['class'] for d in detections]
            if true_class in detected_classes:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation results: {correct}/{total} correct predictions ({accuracy:.2%})")
    return accuracy

def main():
    # Initialize detector
    detector = AnimalHumanDetector()
    
    # Example usage for image
    image_path = r"C:\Users\pytorch\Desktop\assignment\Test_dataset\360_F_77518136_F88I0v3R2mZsKEgxxXMc4iqXlOjK8OLE.jpg"  # Replace with your test image
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            detections, detected_type, processed_image = detector.process_image(image)
            
            # Display results
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Detected type: {detected_type}")
            plt.axis('off')
            plt.show()
            
            # Print detections
            print(f"Detections: {len(detections)}")
            for i, detection in enumerate(detections):
                print(f"{i+1}. {detection['class']} ({detection['confidence']:.2f})")
                
            # Save the processed image with bounding boxes
            output_image_path = "output_detection.jpg"
            cv2.imwrite(output_image_path, processed_image)
            print(f"Processed image saved to {output_image_path}")
    
    # Example usage for video
    video_path = "enter the path" # Replace with your test video
    output_path = "output2_video.mp4"  # Output video path
    if os.path.exists(video_path):
        detections_summary, detected_type = detector.process_video(video_path, output_path)
        print(f"Video processing complete. Dominant type: {detected_type}")
        print(f"Detections summary: {detections_summary}")
    
    # To evaluate on a test dataset:
    # test_data_path = "path/to/test/dataset"  # Each subdirectory is a class
    # accuracy = evaluate_model(detector, test_data_path)
    # if accuracy >= 0.8:
    #     print("✅ Model meets the accuracy requirement (≥80%)")
    # else:
    #     print("❌ Model does not meet the accuracy requirement (≥80%)")

if __name__ == "__main__":
    main()