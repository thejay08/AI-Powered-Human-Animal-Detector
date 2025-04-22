import os
import cv2
import numpy as np
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import csv
import sys

# Import the AnimalHumanDetector class
# Make sure the optimized detector file is in the same directory or update the import path
from parallel_detection import AnimalHumanDetector

class DatasetValidator:
    def __init__(self, dataset_path, output_dir="validation_results"):
        """
        Initialize the validator with dataset path and output directory
        
        Args:
            dataset_path (str): Path to the dataset directory
            output_dir (str): Directory to save validation results
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.detector = AnimalHumanDetector()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.all_results = []
        self.all_true_labels = []
        self.all_pred_labels = []
        self.all_confidences = []
        
        # Video format support
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        # Set frame sampling rate for video processing
        self.frame_sample_rate = 5  # Process every 5th frame
        
        # Maximum number of frames to process per video
        self.max_frames_per_video = 50
        
        # Number of parallel processes for video analysis
        self.max_workers = min(os.cpu_count(), 4)  # Limit to 4 or CPU count, whichever is lower
        
        # Map our class names to detector categories
        self.class_mapping = {
            'cat': 'cat',
            'dog': 'dog', 
            'elephant': 'elephant',
            'lion': 'lion',
            'Person': 'person'
        }
        
        # Target classes - derive from the dataset structure
        self.target_classes = self._get_target_classes()
        print(f"Target classes found: {self.target_classes}")

    def _get_target_classes(self):
        """Get target classes from dataset directory structure"""
        try:
            return [d for d in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, d))]
        except Exception as e:
            print(f"Error reading dataset directory: {e}")
            return []

    def process_video(self, video_path, true_class):
        """
        Process a single video and return detection results
        
        Args:
            video_path (str): Path to the video file
            true_class (str): True class label of the video
            
        Returns:
            dict: Detection results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                return {
                    'path': video_path,
                    'true_class': true_class,
                    'predicted_class': 'error',
                    'confidence': 0.0,
                    'status': 'error',
                    'error': 'Failed to open video file'
                }
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling
            process_every = max(1, fps // self.frame_sample_rate)
            max_frames = min(frame_count, self.max_frames_per_video * process_every)
            
            # Store detections
            all_detections = []
            frames_processed = 0
            current_frame = 0
            
            # Process frames
            while cap.isOpened() and current_frame < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every n-th frame
                if current_frame % process_every == 0:
                    # Resize frame to speed up processing
                    frame = self.detector.smart_resize(frame)
                    
                    # Detect objects - fix for unpacking error
                    try:
                        # Try to handle different return formats
                        result = self.detector.process_image(frame)
                        
                        # Check what was returned and extract detections accordingly
                        if isinstance(result, tuple):
                            if len(result) >= 1:
                                detections = result[0]
                            else:
                                detections = []
                        else:
                            detections = result
                            
                        all_detections.extend(detections if isinstance(detections, list) else [detections])
                    except Exception as e:
                        print(f"Error in frame detection: {e}")
                        # Continue with next frame if there's an error
                        
                    frames_processed += 1
                    if frames_processed >= self.max_frames_per_video:
                        break
                
                current_frame += 1
            
            cap.release()
            
            # Count detections by class
            detection_counts = {}
            for detection in all_detections:
                # Skip if detection is not a dictionary or doesn't have required fields
                if not isinstance(detection, dict) or 'class' not in detection or 'confidence' not in detection:
                    continue
                    
                class_name = detection['class']
                confidence = detection['confidence']
                
                if class_name not in detection_counts:
                    detection_counts[class_name] = []
                    
                detection_counts[class_name].append(confidence)
            
            # Determine the most detected class and average confidence
            most_detected = None
            max_count = 0
            confidence = 0.0
            
            for class_name, confidences in detection_counts.items():
                if len(confidences) > max_count:
                    max_count = len(confidences)
                    most_detected = class_name
                    confidence = sum(confidences) / len(confidences)
            
            # If nothing detected
            if most_detected is None:
                most_detected = "unknown"
                confidence = 0.0
                
            return {
                'path': video_path,
                'true_class': true_class,
                'predicted_class': most_detected,
                'confidence': confidence,
                'frames_processed': frames_processed,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return {
                'path': video_path,
                'true_class': true_class,
                'predicted_class': 'error',
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }

    def validate_dataset(self):
        """Validate the entire dataset and generate metrics"""
        start_time = time.time()
        results = []
        
        # Collect all video paths
        video_paths = []
        true_labels = []
        
        for class_dir in self.target_classes:
            class_path = os.path.join(self.dataset_path, class_dir)
            
            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue
                
            # Get all videos in the class directory
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                
                # Check if it's a video file
                if (os.path.isfile(file_path) and 
                    any(file_path.lower().endswith(ext) for ext in self.video_extensions)):
                    video_paths.append(file_path)
                    true_labels.append(class_dir.lower())
        
        # Process videos in parallel
        print(f"Processing {len(video_paths)} videos using {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Use tqdm for progress tracking
            futures = {executor.submit(self.process_video, path, label): (path, label)
                      for path, label in zip(video_paths, true_labels)}
            
            for future in tqdm(futures, total=len(futures)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    path, label = futures[future]
                    print(f"Failed to process {path}: {e}")
                    results.append({
                        'path': path,
                        'true_class': label,
                        'predicted_class': 'error',
                        'confidence': 0.0,
                        'status': 'error',
                        'error': str(e)
                    })
        
        # Save all results
        self.save_results_to_csv(results)
        
        # Generate metrics based on successful results
        successful_results = [r for r in results if r['status'] == 'success']
        true_classes = [r['true_class'] for r in successful_results]
        pred_classes = [r['predicted_class'] for r in successful_results]
        
        self.generate_metrics(true_classes, pred_classes)
        
        # Print overall statistics
        elapsed_time = time.time() - start_time
        print(f"\nValidation completed in {elapsed_time:.2f} seconds")
        print(f"Processed {len(successful_results)} videos successfully out of {len(results)} total")
        
        return results

    def save_results_to_csv(self, results):
        """Save validation results to CSV file"""
        csv_path = os.path.join(self.output_dir, "validation_results.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['path', 'true_class', 'predicted_class', 'confidence', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                # Filter fields to only include the ones in fieldnames
                filtered_result = {k: result[k] for k in fieldnames if k in result}
                writer.writerow(filtered_result)
        
        print(f"Results saved to {csv_path}")

    def generate_metrics(self, true_classes, pred_classes):
        """Generate and save performance metrics"""
        if not true_classes or not pred_classes:
            print("No data available to generate metrics")
            return
            
        # Get unique classes from both true and predicted
        unique_classes = sorted(list(set(true_classes + pred_classes)))
        
        # Remove 'unknown' and 'error' for confusion matrix
        plot_classes = [c for c in unique_classes if c not in ['unknown', 'error']]
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, pred_classes, labels=plot_classes)
        
        # Save confusion matrix visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=plot_classes, yticklabels=plot_classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Calculate and save classification report
        report = classification_report(true_classes, pred_classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.output_dir, 'classification_report.csv'))
        
        # Calculate overall metrics
        accuracy = accuracy_score(true_classes, pred_classes)
        precision = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
        recall = recall_score(true_classes, pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
        
        # Create and save summary metrics
        summary = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(self.output_dir, 'summary_metrics.csv'), index=False)
        
        # Print summary
        print("\nPerformance Summary:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Create summary visualization
        plt.figure(figsize=(10, 6))
        metrics = list(summary.keys())
        values = list(summary.values())
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = plt.bar(metrics, values, color=colors)
        
        plt.ylim(0, 1.0)
        plt.title('Performance Metrics')
        plt.tight_layout()
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
            
        plt.savefig(os.path.join(self.output_dir, 'metrics_summary.png'))
        plt.close()

def main():
    """Main function to run validation"""
    # Check if dataset path is provided as command line argument
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("Please enter the path to your dataset directory: ")
    
    # Check if output directory is provided
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "validation_results"
    
    # Initialize and run validation
    validator = DatasetValidator(dataset_path, output_dir)
    validator.validate_dataset()
    
    print(f"\nValidation complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()