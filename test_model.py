import os
import cv2
import argparse
from collections import Counter
from parallel_detection import AnimalHumanDetector

def process_dataset(dataset_path, output_dir=None):
    """
    Process all images and videos in a dataset directory
    
    Args:
        dataset_path: Path to directory containing images and videos
        output_dir: Optional path to save processed videos
    """
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist")
        return
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Initialize detector
    detector = AnimalHumanDetector()
    
    # Define supported file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Process each file in the dataset
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Skip directories and unsupported files
        if os.path.isdir(file_path) or (file_ext not in image_extensions and file_ext not in video_extensions):
            continue
        
        print(f"\n--- Processing {filename} ---")
        
        # Process image files
        if file_ext in image_extensions:
            process_image(detector, file_path)
        
        # Process video files
        elif file_ext in video_extensions:
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"processed_{filename}")
            process_video(detector, file_path, output_path)

def process_image(detector, image_path):
    """Process a single image and display results"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Detect objects in image - don't print processing time
    detections, detected_type, detected_animal, _ = detector.process_image(image)
    
    # Generate alert if needed
    if detected_type == "wild":
        print(f"⚠️ ALERT: Wild animal detected - {detected_animal}")
    elif detected_type == "person":
        print(f"👤 ALERT: Person detected")
    else:
        print(f"Detected: {detected_animal} ({detected_type})")

def process_video(detector, video_path, output_path=None):
    """Process video and display results"""
    # Process video with minimal printing (passing None for output_path if not saving)
    # Use a lower sample rate for faster processing
    results = detector.process_video(video_path, output_path, sample_rate=5)
    
    # Extract results
    detected_type = results['detected_type']
    detected_animal = results['detected_animal']
    
    # Print only the final result and alert
    print(f"\n--- Video Analysis Results ---")
    print(f"Dominant detection: {detected_animal} ({detected_type})")
    
    # Generate alert based on detected type
    if detected_type == "wild":
        print(f"⚠️ ALERT: Wild animal detected throughout video - {detected_animal}")
    elif detected_type == "person":
        print(f"👤 ALERT: Person detected throughout video")
    
    # Show distribution of detections (sorted by confidence)
    print("\nDetection distribution:")
    animal_summary = results['animal_summary']
    
    # Sort by confidence score
    sorted_detections = sorted(animal_summary.items(), key=lambda x: x[1], reverse=True)
    for animal, confidence in sorted_detections[:3]:  # Show top 3
        print(f"- {animal}: {confidence:.1f}")

def main():
    parser = argparse.ArgumentParser(description="Test animal and human detection model on a dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", help="Path to save processed videos (optional)")
    args = parser.parse_args()
    
    # Process the dataset
    process_dataset(args.dataset, args.output)
    
if __name__ == "__main__":
    main()
