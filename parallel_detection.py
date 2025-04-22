import cv2
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class AnimalHumanDetector:
    # [keeping all the initialization and other methods the same]
    def __init__(self):
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        
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
        
        # Create custom preprocessing pipeline for better memory management
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        # Create text features for all categories at once (more efficient)
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
        self.confidence_threshold = 0.7
        self.overlap_threshold = 0.5
        
        # Batch processing parameters
        self.batch_size = 64  # Adjust based on GPU memory
        
        # Create Non-Maximum Suppression parameters
        self.nms_threshold = 0.4
        
        # Max workers for parallel processing
        self.max_workers = 4

    # [Keeping all other methods the same]
    def smart_resize(self, image, max_size=1920):
        """Resize image while keeping aspect ratio if larger than max_size"""
        height, width = image.shape[:2]
        
        # Calculate scaling factor to maintain aspect ratio
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        
        return image

    def generate_regions(self, image):
        """Generate region proposals using more efficient approach"""
        height, width = image.shape[:2]
        regions = []
        
        # 1. Use a sliding window with different aspect ratios and scales
        scales = [0.5, 0.75, 1.0]
        aspect_ratios = [0.5, 0.75, 1.0, 1.33, 2.0]
        
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                # Base window is a percentage of image size
                base_size = int(min(height, width) * scale)
                win_h = int(base_size * aspect_ratio)
                win_w = int(base_size / aspect_ratio)
                
                # Skip if window is too large
                if win_h > height or win_w > width:
                    continue
                
                # Calculate steps (stride is 50% of window size)
                stride_h = max(win_h // 2, 1)
                stride_w = max(win_w // 2, 1)
                
                # Generate positions
                for y in range(0, height - win_h + 1, stride_h):
                    for x in range(0, width - win_w + 1, stride_w):
                        regions.append((x, y, x + win_w, y + win_h))
        
        # 2. Add whole image and quadrants for coverage
        regions.append((0, 0, width, height))  # Whole image
        
        # Add quadrants
        half_h, half_w = height // 2, width // 2
        regions.append((0, 0, half_w, half_h))  # Top-left
        regions.append((half_w, 0, width, half_h))  # Top-right
        regions.append((0, half_h, half_w, height))  # Bottom-left
        regions.append((half_w, half_h, width, height))  # Bottom-right
        
        # 3. Add grid-based regions
        grid_sizes = [2, 3]
        for grid_size in grid_sizes:
            cell_height = height // grid_size
            cell_width = width // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x1 = j * cell_width
                    y1 = i * cell_height
                    x2 = min(x1 + cell_width, width)
                    y2 = min(y1 + cell_height, height)
                    
                    regions.append((x1, y1, x2, y2))
        
        # Remove duplicates and filter very small regions
        filtered_regions = []
        min_area = (width * height) / 100  # 1% of image area
        
        seen = set()
        for region in regions:
            region_tuple = tuple(region)
            if region_tuple in seen:
                continue
                
            x1, y1, x2, y2 = region
            if (x2 - x1) * (y2 - y1) >= min_area:
                filtered_regions.append(region)
                seen.add(region_tuple)
        
        return filtered_regions

    def preprocess_batch(self, regions, image):
        """Preprocess a batch of image regions for CLIP"""
        processed_images = []
        valid_regions = []
        
        for x1, y1, x2, y2 in regions:
            try:
                # Extract region
                region = image[y1:y2, x1:x2]
                
                # Skip if region is too small
                if region.shape[0] < 10 or region.shape[1] < 10:
                    continue
                
                # Convert to PIL image
                pil_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                
                # Apply preprocessing
                processed = self.preprocess(pil_image)
                processed_images.append(processed)
                valid_regions.append((x1, y1, x2, y2))
            except Exception as e:
                continue
                
        if not processed_images:
            return None, None
            
        # Stack tensors into a batch
        image_batch = torch.stack(processed_images).to(self.device)
        
        return image_batch, valid_regions

    def process_image_batch(self, image_batch, valid_regions):
        """Process a batch of preprocessed images with CLIP"""
        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
            # Get top predictions
            values, indices = similarity.topk(1, dim=-1)
            
            detections = []
            for i in range(len(valid_regions)):
                confidence = values[i][0].item()
                class_idx = indices[i][0].item()
                
                if confidence >= self.confidence_threshold:
                    class_name = list(self.categories.keys())[class_idx]
                    animal_type = self.categories[class_name]
                    x1, y1, x2, y2 = valid_regions[i]
                    
                    detections.append({
                        'class': class_name,
                        'type': animal_type,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            return detections

    def non_maximum_suppression(self, detections):
        """Apply NMS to remove overlapping detections"""
        if not detections:
            return []
            
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        kept_detections = []
        
        while detections:
            best = detections.pop(0)
            kept_detections.append(best)
            
            # Filter out overlapping boxes
            filtered_detections = []
            best_x1, best_y1, best_x2, best_y2 = best['bbox']
            best_area = (best_x2 - best_x1) * (best_y2 - best_y1)
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                
                # Calculate intersection
                xx1 = max(best_x1, x1)
                yy1 = max(best_y1, y1)
                xx2 = min(best_x2, x2)
                yy2 = min(best_y2, y2)
                
                intersection = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                
                # Calculate union
                detection_area = (x2 - x1) * (y2 - y1)
                union = best_area + detection_area - intersection
                
                # Calculate IoU
                iou = intersection / union if union > 0 else 0
                
                # Keep if IoU is below threshold
                if iou < self.nms_threshold:
                    filtered_detections.append(detection)
            
            detections = filtered_detections
            
        return kept_detections

    def detect_objects(self, image):
        """Detect humans and animals in the image using batched processing"""
        start_time = time.time()
        original_image = image.copy()
        
        # Resize image if it's too large
        image = self.smart_resize(image)
        
        # Get region proposals
        regions = self.generate_regions(image)
        print(f"Generated {len(regions)} regions in {time.time() - start_time:.2f} seconds")
        
        # Process regions in batches
        all_detections = []
        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]
            image_batch, valid_regions = self.preprocess_batch(batch_regions, image)
            
            if image_batch is not None:
                batch_detections = self.process_image_batch(image_batch, valid_regions)
                all_detections.extend(batch_detections)
        
        # Apply NMS
        final_detections = self.non_maximum_suppression(all_detections)
        
        # Scale detections back to original image size if it was resized
        if image.shape != original_image.shape:
            scale_x = original_image.shape[1] / image.shape[1]
            scale_y = original_image.shape[0] / image.shape[0]
            
            for detection in final_detections:
                x1, y1, x2, y2 = detection['bbox']
                detection['bbox'] = [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]
        
        processing_time = time.time() - start_time
        print(f"Total detection time: {processing_time:.2f} seconds")
        
        return final_detections, original_image

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
        
        # Detect objects
        detections, original_image = self.detect_objects(image)
        
        # Draw detections
        processed_image = self.draw_detections(original_image.copy(), detections)
        
        # Determine dominant type and specific animal classes
        dominant_type = defaultdict(float)
        animal_counts = defaultdict(float)
        
        for detection in detections:
            # Track type frequencies
            dominant_type[detection['type']] += detection['confidence']
            # Track specific animal classes
            animal_counts[detection['class']] += detection['confidence']
        
        # Get most frequent type
        detected_type = max(dominant_type.items(), key=lambda x: x[1])[0] if dominant_type else "unknown"
        
        # Get most frequent specific animal
        detected_animal = max(animal_counts.items(), key=lambda x: x[1])[0] if animal_counts else "unknown"
        
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        return detections, detected_type, detected_animal, processed_image

    def parallel_frame_processing(self, frame_batch):
        """Process a batch of frames in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_frame, frame) for frame in frame_batch]
            
            for future in futures:
                results.append(future.result())
                
        return results
    
    def _process_frame(self, frame):
        """Process a single frame for parallel execution"""
        # Use a smaller confidence threshold for video frames
        temp_threshold = self.confidence_threshold
        self.confidence_threshold = 0.6  # Lower threshold for video frames
        
        # Get region proposals
        regions = self.generate_regions(frame)
        
        # Process regions in batches
        all_detections = []
        for i in range(0, len(regions), self.batch_size):
            batch_regions = regions[i:i + self.batch_size]
            image_batch, valid_regions = self.preprocess_batch(batch_regions, frame)
            
            if image_batch is not None:
                batch_detections = self.process_image_batch(image_batch, valid_regions)
                all_detections.extend(batch_detections)
        
        # Apply NMS
        final_detections = self.non_maximum_suppression(all_detections)
        
        # Draw detections
        processed_frame = self.draw_detections(frame.copy(), final_detections)
        
        # Determine dominant type
        dominant_type = defaultdict(float)
        animal_counts = defaultdict(float)
        
        for detection in final_detections:
            dominant_type[detection['type']] += detection['confidence']
            animal_counts[detection['class']] += detection['confidence']
        
        detected_type = max(dominant_type.items(), key=lambda x: x[1])[0] if dominant_type else "unknown"
        detected_animal = max(animal_counts.items(), key=lambda x: x[1])[0] if animal_counts else "unknown"
        
        # Reset threshold
        self.confidence_threshold = temp_threshold
        
        return final_detections, detected_type, detected_animal, processed_frame

    def process_video(self, video_path, output_path=None, sample_rate=10):
        """Process video file and save output with improved GPU batching"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer if path is provided
        out = None
        if output_path:
            # Try different codecs if your system doesn't support mp4v
            try:
                # First attempt with mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                # Test if video writer is working
                if not out.isOpened():
                    raise Exception("Failed to open VideoWriter with mp4v codec")
                    
            except Exception as e:
                print(f"Warning: mp4v codec failed: {e}")
                # Try with other codecs
                for codec in ['XVID', 'MJPG', 'H264', 'avc1']:
                    try:
                        print(f"Trying codec: {codec}")
                        if out is not None and out.isOpened():
                            out.release()  # Release previous attempt
                            
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        test_output_path = f"{os.path.splitext(output_path)[0]}_{codec}.avi"
                        out = cv2.VideoWriter(test_output_path, fourcc, fps, (frame_width, frame_height))
                        
                        if out.isOpened():
                            print(f"Using codec {codec} with output file: {test_output_path}")
                            output_path = test_output_path
                            break
                    except Exception as e2:
                        print(f"Codec {codec} failed: {e2}")
                        continue
                
                if out is None or not out.isOpened():
                    print("ERROR: Could not create VideoWriter with any codec. Will continue processing without saving output.")
        
        # Calculate frames to process based on sample rate
        process_every = max(1, fps // sample_rate)  # Process N frames per second
        
        # Track detections by type and class with confidences
        detections_summary = defaultdict(float)  # For animal types (wild, pet, farm, person)
        animal_counts = defaultdict(float)       # For specific animal classes (dog, cat, lion, etc)
        
        # Store frames with wild animals or people
        alert_frames = []
        
        frame_count = 0
        processed_frames = []
        last_detections = []  # Store previous detections
        detected_type = "unknown"
        detected_animal = "unknown"
        
        start_time = time.time()
        
        print(f"Processing video with {fps} fps, sampling every {process_every} frames")
        print(f"Total frames: {total_frames}")
        
        # Prepare batches for parallel processing
        frame_batch = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize frame to speed up processing
            frame = self.smart_resize(frame)
            
            # Process key frames
            if frame_count % process_every == 0:
                frame_batch.append(frame)
                
                # Process batch when it reaches desired size
                if len(frame_batch) >= self.max_workers:
                    results = self.parallel_frame_processing(frame_batch)
                    
                    for i, (detections, det_type, det_animal, processed_frame) in enumerate(results):
                        # Add to summary
                        for detection in detections:
                            detections_summary[detection['type']] += detection['confidence']
                            animal_counts[detection['class']] += detection['confidence']
                        
                        # Record frame number for alerts rather than printing immediately
                        if det_type == 'wild' or det_type == 'person':
                            current_frame_num = frame_count - len(frame_batch) + i + 1
                            alert_frames.append((current_frame_num, det_type, det_animal))
                        
                        # Save processed frame
                        processed_frames.append(processed_frame)
                        
                        # Update last detections
                        if i == len(results) - 1:
                            last_detections = detections
                            detected_type = det_type
                            detected_animal = det_animal
                    
                    # Check if video writer was created successfully
                    if out is not None and out.isOpened():
                        for processed_frame in processed_frames:
                            # Ensure the frame has the right dimensions
                            if processed_frame.shape[:2] != (frame_height, frame_width):
                                processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))
                            # Write frame to video    
                            out.write(processed_frame)
                    
                    # Clear batches
                    frame_batch = []
                    processed_frames = []
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    progress = frame_count / total_frames * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {elapsed:.1f}s elapsed")
            else:
                # Use previous detections for intermediate frames
                if last_detections:
                    processed_frame = self.draw_detections(frame.copy(), last_detections)
                    if out is not None and out.isOpened():
                        # Ensure the frame has the right dimensions
                        if processed_frame.shape[:2] != (frame_height, frame_width):
                            processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))
                        out.write(processed_frame)
                elif out is not None and out.isOpened():
                    # Ensure the frame has the right dimensions
                    if frame.shape[:2] != (frame_height, frame_width):
                        frame = cv2.resize(frame, (frame_width, frame_height))
                    # No detections yet, just write the original frame
                    out.write(frame)
        
        # Process any remaining frames in the batch
        if frame_batch:
            results = self.parallel_frame_processing(frame_batch)
            for i, (detections, det_type, det_animal, processed_frame) in enumerate(results):
                # Add to summary
                for detection in detections:
                    detections_summary[detection['type']] += detection['confidence']
                    animal_counts[detection['class']] += detection['confidence']
                
                # Write the processed frame to output
                if out is not None and out.isOpened():
                    # Ensure the frame has the right dimensions
                    if processed_frame.shape[:2] != (frame_height, frame_width):
                        processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))
                    out.write(processed_frame)
        
        # Release resources
        cap.release()
        if out is not None and out.isOpened():
            out.release()
            print(f"Output video saved to: {output_path}")
        else:
            print("No output video was saved due to issues with VideoWriter")
        
        # Determine the dominant category based on confidence-weighted detections
        detected_type = max(detections_summary.items(), key=lambda x: x[1])[0] if detections_summary else "unknown"
        detected_animal = max(animal_counts.items(), key=lambda x: x[1])[0] if animal_counts else "unknown"
        
        # Generate alerts at the end
        if alert_frames:
            print("\n===== ALERTS SUMMARY =====")
            for frame_num, alert_type, alert_animal in alert_frames:
                alert_symbol = "⚠️" if alert_type == "wild" else "👤"
                print(f"{alert_symbol} ALERT: {alert_animal} ({alert_type}) detected at frame {frame_num}!")
            print("========================")
        
        if detected_type == 'wild':
            print(f"⚠️ ALERT: Wild animal detected! Specific animal: {detected_animal}")
        elif detected_type == 'person':
            print(f"👤 ALERT: Person detected!")
        
        total_time = time.time() - start_time
        print(f"Total video processing time: {total_time:.2f} seconds")
        print(f"Average processing speed: {frame_count / total_time:.2f} frames/second")
        
        # Return both type-based and animal-specific detection summaries
        return {
            'type_summary': dict(detections_summary),
            'animal_summary': dict(animal_counts),
            'detected_type': detected_type,
            'detected_animal': detected_animal,
            'output_path': output_path if out is not None and out.isOpened() else None
        }

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
            detections, detected_type, detected_animal, _ = detector.process_image(image)
            
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
    image_path = r"C:\Animal_detaction\animal_and_person_detection\Screenshot 2025-04-22 174233.png"  # Replace with your test image
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            detections, detected_type, detected_animal, processed_image = detector.process_image(image)
            
            # Display results
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Detected: {detected_animal} ({detected_type})")
            plt.axis('off')
            plt.show()
            
            # Print detections
            print(f"Detections: {len(detections)}")
            for i, detection in enumerate(detections):
                print(f"{i+1}. {detection['class']} ({detection['confidence']:.2f})")
    
    # Example usage for video
    video_path = "C:\Animal_detaction\animal_and_person_detection\Dataset\Cat\853757-hd_1920_1080_25fps.mp4"  # Replace with your test video
    output_path = "output_video.mp4"
    if os.path.exists(video_path):
        results = detector.process_video(video_path, output_path)
        
        # Access results with more detail
        detected_type = results['detected_type']
        detected_animal = results['detected_animal']
        actual_output_path = results['output_path']
        
        print(f"Video processing complete. Dominant animal: {detected_animal} ({detected_type})")
        print(f"Type summary: {results['type_summary']}")
        print(f"Animal summary: {results['animal_summary']}")
        if actual_output_path:
            print(f"Output video saved to: {actual_output_path}")
        else:
            print("No output video was generated")
    
    # To evaluate on a test dataset:
    # test_data_path = "path/to/test/dataset"  # Each subdirectory is a class
    # accuracy = evaluate_model(detector, test_data_path)
    # if accuracy >= 0.8:
    #     print("✅ Model meets the accuracy requirement (≥80%)")
    # else:
    #     print("❌ Model does not meet the accuracy requirement (≥80%)")

if __name__ == "__main__":
    main()