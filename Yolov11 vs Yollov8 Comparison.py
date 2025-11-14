import time
import cv2
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def video_comparison(video_path):
    """Compare YOLOv8 and YOLOv11 using your drone video"""
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please check the file path and make sure the video file exists.")
        return
    
    # Load models (they'll download automatically if not present)
    models = {
        'YOLOv8n': YOLO('yolov8n.pt'),
        'YOLOv11n': YOLO('yolo11n.pt')
    }
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps_original
    
    print(f"Video info:")
    print(f"  Original FPS: {fps_original:.2f}")
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Extract sample frames for testing (every 30th frame to get good variety)
    sample_frames = []
    frame_indices = []
    
    frame_idx = 0
    while len(sample_frames) < 20 and frame_idx < frame_count:  # Get up to 20 sample frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            sample_frames.append(frame)
            frame_indices.append(frame_idx)
        frame_idx += max(1, frame_count // 20)  # Sample evenly across video
    
    cap.release()
    
    print(f"Extracted {len(sample_frames)} sample frames for testing")
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Warmup with first frame
        if sample_frames:
            _ = model.predict(sample_frames[0], verbose=False)
        
        # Speed test on sample frames
        times = []
        detections_per_frame = []
        confidence_scores = []
        
        for frame in sample_frames:
            start = time.time()
            pred_results = model.predict(frame, verbose=False)
            inference_time = time.time() - start
            times.append(inference_time)
            
            # Count detections and get confidence scores
            if pred_results and len(pred_results) > 0:
                boxes = pred_results[0].boxes
                if boxes is not None:
                    detections_per_frame.append(len(boxes))
                    # Get confidence scores
                    if hasattr(boxes, 'conf') and boxes.conf is not None:
                        frame_confidences = boxes.conf.cpu().numpy()
                        confidence_scores.extend(frame_confidences)
                else:
                    detections_per_frame.append(0)
            else:
                detections_per_frame.append(0)
        
        avg_time = np.mean(times)
        fps_capability = 1 / avg_time
        avg_detections = np.mean(detections_per_frame)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Try to get model info (without requiring COCO dataset)
        try:
            # Get model parameters count
            total_params = sum(p.numel() for p in model.model.parameters())
            model_size_mb = total_params * 4 / (1024 * 1024)  # Rough estimate (4 bytes per parameter)
        except:
            total_params = "Unknown"
            model_size_mb = "Unknown"
        
        results[name] = {
            'avg_time_ms': avg_time * 1000,
            'fps_capability': fps_capability,
            'avg_detections': avg_detections,
            'avg_confidence': avg_confidence,
            'total_frames_tested': len(sample_frames),
            'total_params': total_params,
            'model_size_mb': model_size_mb
        }
        
        print(f"  Average inference time: {avg_time*1000:.2f} ms")
        print(f"  Processing capability: {fps_capability:.2f} FPS")
        print(f"  Average detections per frame: {avg_detections:.1f}")
        print(f"  Average confidence score: {avg_confidence:.3f}")
        
        # Real-time capability assessment
        can_process_realtime = fps_capability >= fps_original
        print(f"  Can process this video in real-time: {'Yes' if can_process_realtime else 'No'}")
    
    # Create comprehensive comparison plots
    create_comparison_plots(results, fps_original)
    
    # Process a short segment with visualizations
    process_video_segment(video_path, models, start_time=10, duration=5)
    
    return results

def create_comparison_plots(results, original_fps):
    """Create detailed comparison plots with thesis-quality formatting"""
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white'
    })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    models_names = list(results.keys())
    fps_values = [results[m]['fps_capability'] for m in models_names]
    time_values = [results[m]['avg_time_ms'] for m in models_names]
    detection_values = [results[m]['avg_detections'] for m in models_names]
    confidence_values = [results[m]['avg_confidence'] for m in models_names]
    
    # Professional color scheme
    colors = ['#2E86AB', '#A23B72']  # Blue and burgundy
    edge_colors = ['#1F5F82', '#7A2B56']
    
    # Plot 1: Processing Speed Capability
    bars1 = ax1.bar(models_names, fps_values, color=colors, alpha=0.8, 
                    edgecolor=edge_colors, linewidth=1.5, width=0.6)
    ax1.axhline(y=original_fps, color='#F18F01', linestyle='--', linewidth=2, 
                label=f'Original Video FPS ({original_fps:.1f})')
    ax1.set_title('(a) Processing Speed Capability', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Frames Per Second (FPS)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels with better positioning
    for bar, fps in zip(bars1, fps_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + max(fps_values)*0.02, 
                f'{fps:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Set y-axis limits with padding
    ax1.set_ylim(0, max(fps_values) * 1.15)
    
    # Plot 2: Inference Time per Frame
    bars2 = ax2.bar(models_names, time_values, color=colors, alpha=0.8,
                    edgecolor=edge_colors, linewidth=1.5, width=0.6)
    ax2.set_title('(b) Inference Time per Frame', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Processing Time (ms)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, time_ms in zip(bars2, time_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + max(time_values)*0.02, 
                f'{time_ms:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylim(0, max(time_values) * 1.15)
    
    # Plot 3: Average Detections per Frame
    bars3 = ax3.bar(models_names, detection_values, color=colors, alpha=0.8,
                    edgecolor=edge_colors, linewidth=1.5, width=0.6)
    ax3.set_title('(c) Detection Performance', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('Average Detections per Frame', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, det in zip(bars3, detection_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + max(detection_values)*0.02, 
                f'{det:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3.set_ylim(0, max(detection_values) * 1.15 if max(detection_values) > 0 else 1)
    
    # Plot 4: Average Confidence Scores
    bars4 = ax4.bar(models_names, confidence_values, color=colors, alpha=0.8,
                    edgecolor=edge_colors, linewidth=1.5, width=0.6)
    ax4.set_title('(d) Detection Confidence', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Average Confidence Score', fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, conf in zip(bars4, confidence_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Main title
    fig.suptitle('Performance Comparison: YOLOv8 vs YOLOv11 on Drone Video Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save high-quality images for thesis
    plt.savefig('thesis_yolo_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('thesis_yolo_comparison.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('thesis_yolo_comparison.eps', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("Thesis-quality plots saved as:")
    print("  - thesis_yolo_comparison.png (for digital use)")
    print("  - thesis_yolo_comparison.pdf (for LaTeX)")
    print("  - thesis_yolo_comparison.eps (for publication)")
    
    plt.show()

def create_detailed_performance_chart(results, detection_analysis):
    """Create a comprehensive performance comparison chart"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    models_names = list(results.keys())
    
    # Extract data
    fps_values = [results[m]['fps_capability'] for m in models_names]
    time_values = [results[m]['avg_time_ms'] for m in models_names]
    total_detections = [detection_analysis[m]['total_detections'] for m in models_names]
    unique_classes = [detection_analysis[m]['unique_classes'] for m in models_names]
    
    colors = ['#2E86AB', '#A23B72']
    
    # Performance efficiency plot
    ax1.scatter(time_values, fps_values, s=200, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
    for i, model in enumerate(models_names):
        ax1.annotate(model, (time_values[i], fps_values[i]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Inference Time (ms)', fontsize=12)
    ax1.set_ylabel('Processing Capability (FPS)', fontsize=12)
    ax1.set_title('(a) Speed vs Efficiency Trade-off', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Total detections comparison
    bars2 = ax2.bar(models_names, total_detections, color=colors, alpha=0.8, width=0.6)
    ax2.set_title('(b) Total Objects Detected', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Total Detections', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, total in zip(bars2, total_detections):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + max(total_detections)*0.02, 
                f'{total}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Object diversity
    bars3 = ax3.bar(models_names, unique_classes, color=colors, alpha=0.8, width=0.6)
    ax3.set_title('(c) Object Class Diversity', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('Unique Object Classes Detected', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, classes in zip(bars3, unique_classes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                f'{classes}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Model parameters comparison
    param_values = [results[m]['total_params'] for m in models_names if results[m]['total_params'] != "Unknown"]
    if param_values:
        param_labels = [m for m in models_names if results[m]['total_params'] != "Unknown"]
        param_millions = [p/1e6 for p in param_values]
        
        bars4 = ax4.bar(param_labels, param_millions, color=colors[:len(param_labels)], alpha=0.8, width=0.6)
        ax4.set_title('(d) Model Complexity', fontsize=14, fontweight='bold', pad=20)
        ax4.set_ylabel('Parameters (Millions)', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, params in zip(bars4, param_millions):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + max(param_millions)*0.02, 
                    f'{params:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'Model Parameters\nData Unavailable', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('(d) Model Complexity', fontsize=14, fontweight='bold', pad=20)
    
    fig.suptitle('Detailed Performance Analysis: YOLOv8 vs YOLOv11', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save detailed analysis plots
    plt.savefig('thesis_detailed_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('thesis_detailed_analysis.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("Detailed analysis plots saved as:")
    print("  - thesis_detailed_analysis.png")
    print("  - thesis_detailed_analysis.pdf")
    
    plt.show()

def process_video_segment(video_path, models, start_time=10, duration=5):
    """Process a short video segment and save results"""
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set start position
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    frame_count = 0
    max_frames = int(fps * duration)
    
    print(f"\nProcessing {duration}s segment starting at {start_time}s...")
    
    for model_name, model in models.items():
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # Reset position
        
        # Create output video writer
        output_path = f'{model_name}_drone_detection_sample.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        ret, first_frame = cap.read()
        if ret:
            height, width = first_frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process first frame
            results = model.predict(first_frame, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frame_count = 1
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = model.predict(frame, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                
                frame_count += 1
            
            out.release()
            print(f"  Saved {model_name} results to: {output_path}")
    
    cap.release()

def analyze_detection_performance(video_path, models, sample_size=50):
    """Detailed analysis of detection performance on video frames"""
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly across the video
    frame_indices = np.linspace(0, frame_count-1, sample_size, dtype=int)
    
    detection_analysis = {}
    
    for model_name, model in models.items():
        print(f"\nDetailed analysis for {model_name}...")
        
        detections_by_class = {}
        all_confidences = []
        processing_times = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            start_time = time.time()
            results = model.predict(frame, verbose=False, conf=0.25)  # Lower confidence threshold
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    # Get class names and confidences
                    if hasattr(boxes, 'cls') and hasattr(boxes, 'conf'):
                        classes = boxes.cls.cpu().numpy()
                        confidences = boxes.conf.cpu().numpy()
                        
                        for cls, conf in zip(classes, confidences):
                            class_name = model.names[int(cls)]
                            if class_name not in detections_by_class:
                                detections_by_class[class_name] = []
                            detections_by_class[class_name].append(conf)
                            all_confidences.append(conf)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{sample_size} frames...")
        
        detection_analysis[model_name] = {
            'detections_by_class': detections_by_class,
            'avg_processing_time': np.mean(processing_times),
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
            'total_detections': len(all_confidences),
            'unique_classes': len(detections_by_class)
        }
        
        print(f"  Found {len(detections_by_class)} different object classes")
        print(f"  Total detections: {len(all_confidences)}")
        print(f"  Average confidence: {np.mean(all_confidences):.3f}" if all_confidences else "  No detections")
    
    cap.release()
    return detection_analysis

def print_summary(results, detection_analysis=None):
    """Print detailed summary"""
    print(f"\n{'='*60}")
    print("DETAILED COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for name in results.keys():
        r = results[name]
        print(f"\n{name}:")
        print(f"  Processing Speed: {r['fps_capability']:.1f} FPS")
        print(f"  Inference Time: {r['avg_time_ms']:.1f} ms per frame")
        print(f"  Avg detections per frame: {r['avg_detections']:.1f}")
        print(f"  Average confidence: {r['avg_confidence']:.3f}")
        print(f"  Frames tested: {r['total_frames_tested']}")
        if r['total_params'] != "Unknown":
            print(f"  Model parameters: {r['total_params']:,}")
    
    if detection_analysis:
        print(f"\n{'='*60}")
        print("DETECTION ANALYSIS")
        print(f"{'='*60}")
        
        for model_name, analysis in detection_analysis.items():
            print(f"\n{model_name} - Detected Object Classes:")
            for class_name, confidences in analysis['detections_by_class'].items():
                avg_conf = np.mean(confidences)
                count = len(confidences)
                print(f"  {class_name}: {count} detections (avg confidence: {avg_conf:.3f})")
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    models_list = list(results.keys())
    if len(models_list) == 2:
        model1, model2 = models_list
        
        speed_diff = results[model2]['fps_capability'] / results[model1]['fps_capability']
        detection_diff = results[model2]['avg_detections'] / max(results[model1]['avg_detections'], 0.1)
        
        print(f"{model2} vs {model1}:")
        print(f"  Speed ratio: {speed_diff:.2f}x ({'faster' if speed_diff > 1 else 'slower'})")
        print(f"  Detection ratio: {detection_diff:.2f}x ({'more' if detection_diff > 1 else 'fewer'} detections)")

if __name__ == "__main__":
    # Your video path
    video_path = r"c:\Users\Josh\OneDrive\josh's folder\Curtin University\Thesis\Videos\Test video"
    
    # Try common video extensions
    video_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV']
    
    found_video = False
    for ext in video_extensions:
        test_path = video_path + ext
        if os.path.exists(test_path):
            print(f"Found video file: {test_path}")
            
            # Run basic comparison
            results = video_comparison(test_path)
            
            if results:
                # Run detailed detection analysis
                models = {
                    'YOLOv8n': YOLO('yolov8n.pt'),
                    'YOLOv11n': YOLO('yolo11n.pt')
                }
                detection_analysis = analyze_detection_performance(test_path, models)
                
                # Create detailed performance chart
                create_detailed_performance_chart(results, detection_analysis)
                
                # Print comprehensive summary
                print_summary(results, detection_analysis)
                
            found_video = True
            break
    
    if not found_video:
        print(f"Video file not found at: {video_path}")
        print("Please check the file path and extension (.mp4, .avi, .mov, etc.)")
        print("\nAlternatively, you can specify the exact file path:")
        print('video_path = r"your_exact_video_path_with_extension.mp4"')