"""
This script implements a comparative analysis of object classification performance
using LiDAR, RADAR, and LiDAR+RADAR fusion approach on the TruckScenes dataset.
The system employs k-fold cross-validation to evaluate classification metrics
including accuracy, precision, recall, F1-score, balanced accuracy, and specificity.
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler, label_binarize
import open3d as o3d
import warnings
from tqdm import tqdm
from truckscenes import TruckScenes
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', module='open3d')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Tight layout not applied*')

# Configuration constants
DATASET_ROOT_PATH = r'/Users/lorenzo/Assignment-3/man-truckscenes'
DATASET_VERSION = 'v1.0-mini'
CROSS_VALIDATION_FOLDS = 5
RANDOM_SEED = 42
MAX_POINTS_PER_OBJECT = 50  # Limit points to prevent overfitting
GAUSSIAN_NOISE_LEVEL = 0.03  # Small noise to improve generalization

print("Specifications:")
print(f"- Cross-validation: {CROSS_VALIDATION_FOLDS}-fold")
print(f"- Classifier: Random Forest")
print(f"- Feature normalization: StandardScaler")

def load_point_cloud_data(file_path):
    """
    Load point cloud data from PCD file format.

    Args:
        file_path (str): Path to the .pcd file
        
    Returns:
        numpy.ndarray: Point cloud coordinates as (N, 3) array or empty array if loading fails
    """
    try:
        point_cloud = o3d.io.read_point_cloud(file_path)
        return np.asarray(point_cloud.points)
    except Exception:
        return np.empty((0, 3))


def add_gaussian_noise(feature_vector, noise_standard_deviation=0.1):
    """
    Add Gaussian noise to feature vectors to improve model generalization.
    
    Args:
        feature_vector (numpy.ndarray): Input feature vector
        noise_standard_deviation (float): Standard deviation of Gaussian noise
        
    Returns:
        numpy.ndarray: Feature vector with added noise
    """
    noise = np.random.normal(0, noise_standard_deviation, feature_vector.shape)
    return feature_vector + noise


def extract_geometric_features(annotation_data):
    """
    Extract geometric features from object annotations.
    
    Args:
        annotation_data (dict): Object annotation containing size and translation
        
    Returns:
        numpy.ndarray: Array of 9 geometric features
    """
    object_size = annotation_data.get('size', [1, 1, 1])
    object_position = annotation_data.get('translation', [0, 0, 0])
    
    # Basic geometric measurements
    length_dimension = max(object_size[0], object_size[1])
    width_dimension = min(object_size[0], object_size[1])
    height_dimension = object_size[2]
    
    # Normalized geometric ratios
    aspect_ratio = length_dimension / (width_dimension + 1e-6)
    height_width_ratio = height_dimension / (width_dimension + 1e-6)
    volume_logarithmic = np.log(length_dimension * width_dimension * height_dimension + 1)
    
    # Discretized spatial position (reduces overfitting to exact locations)
    utm_offset_x = 571700  # UTM coordinate normalization
    utm_offset_y = 5367800
    utm_offset_z = 594
    
    position_bucket_x = int((object_position[0] - utm_offset_x) / 50) % 10
    position_bucket_y = int((object_position[1] - utm_offset_y) / 50) % 10
    position_bucket_z = int((object_position[2] - utm_offset_z) / 2) % 5
    
    return np.array([
        length_dimension, width_dimension, height_dimension,
        aspect_ratio, height_width_ratio, volume_logarithmic,
        position_bucket_x, position_bucket_y, position_bucket_z
    ])


def extract_sensor_specific_features(point_cloud_data, annotation_data, sensor_modality):
    """
    Extract sensor-specific features from point cloud data.
    
    Args:
        point_cloud_data (numpy.ndarray): Point cloud coordinates
        annotation_data (dict): Object annotation data
        sensor_modality (str): Sensor type ("LIDAR", "RADAR", "LIDAR+RADAR")
        
    Returns:
        numpy.ndarray: Sensor-specific feature vector
    """
    if point_cloud_data.size == 0:
        if sensor_modality == "LIDAR":
            return np.zeros(6)  # LIDAR-specific feature count
        elif sensor_modality == "RADAR":
            return np.zeros(5)  # RADAR-specific feature count
        else:
            return np.zeros(4)
    
    # Subsample points to prevent discriminative overfitting
    if len(point_cloud_data) > MAX_POINTS_PER_OBJECT:
        selected_indices = np.random.choice(len(point_cloud_data), MAX_POINTS_PER_OBJECT, replace=False)
        point_cloud_data = point_cloud_data[selected_indices]
    
    # Calculate basic point cloud statistics
    point_count_limited = min(len(point_cloud_data), MAX_POINTS_PER_OBJECT)
    object_volume = np.prod(annotation_data.get('size', [1, 1, 1]))
    point_density = point_count_limited / object_volume
    
    if len(point_cloud_data) > 1:
        centroid_position = np.mean(point_cloud_data, axis=0)
        distances_from_centroid = np.linalg.norm(point_cloud_data - centroid_position, axis=1)
        spatial_spread = np.std(distances_from_centroid)
        spatial_coverage = len(np.unique(np.round(point_cloud_data, 1))) / len(point_cloud_data)
        
        # Sensor-specific feature extraction
        if sensor_modality == "LIDAR":
            height_variance = np.var(point_cloud_data[:, 2])
            surface_roughness = (np.std(distances_from_centroid) / np.mean(distances_from_centroid) if np.mean(distances_from_centroid) > 0 else 0)
            return np.array([point_count_limited, point_density, spatial_spread, spatial_coverage, height_variance, surface_roughness])
            
        elif sensor_modality == "RADAR":
            penetration_capability = point_count_limited / (object_volume + 1)
            return np.array([point_count_limited, point_density, spatial_spread, spatial_coverage, penetration_capability])
    else:
        # Handle single-point cases
        if sensor_modality == "LIDAR":
            return np.array([point_count_limited, point_density, 0, 0, 0, 0])
        elif sensor_modality == "RADAR":
            return np.array([point_count_limited, point_density, 0, 0, 0])
    
    # Default case
    return np.array([point_count_limited, point_density, spatial_spread, spatial_coverage])


def extract_fusion_sensor_features(lidar_points, radar_points, annotation_data):
    """
    Extract features for LIDAR+RADAR sensor fusion.
    
    Args:
        lidar_points (numpy.ndarray): LIDAR point cloud data
        radar_points (numpy.ndarray): RADAR point cloud data
        annotation_data (dict): Object annotation data
        
    Returns:
        numpy.ndarray: Array of 7 fusion-specific features
    """
    # Normalize coordinates to vehicle reference frame
    object_center = np.array(annotation_data['translation'])
    utm_normalized_center = np.array([
        (object_center[0] - 571700) / 100,
        (object_center[1] - 5367800) / 100,
        object_center[2] - 594
    ])
    object_size = np.array(annotation_data['size'])
    bounding_box_min = utm_normalized_center - object_size / 2
    bounding_box_max = utm_normalized_center + object_size / 2
    
    # Filter points within object bounding box for each sensor
    if lidar_points.size > 0:
        lidar_in_bbox = lidar_points[np.all((lidar_points >= bounding_box_min) & (lidar_points <= bounding_box_max), axis=1)]
    else:
        lidar_in_bbox = np.empty((0, 3))
    
    if radar_points.size > 0:
        radar_in_bbox = radar_points[np.all((radar_points >= bounding_box_min) & (radar_points <= bounding_box_max), axis=1)]
    else:
        radar_in_bbox = np.empty((0, 3))
    
    # Calculate detection statistics
    lidar_detection_count = len(lidar_in_bbox)
    radar_detection_count = len(radar_in_bbox)
    total_detections = lidar_detection_count + radar_detection_count
    
    # Sensor contribution ratios
    if total_detections > 0:
        lidar_contribution_ratio = lidar_detection_count / total_detections
        radar_contribution_ratio = radar_detection_count / total_detections
        detection_balance_score = (min(lidar_contribution_ratio, radar_contribution_ratio) / max(lidar_contribution_ratio, radar_contribution_ratio) if max(lidar_contribution_ratio, radar_contribution_ratio) > 0 else 0)
    else:
        lidar_contribution_ratio = radar_contribution_ratio = detection_balance_score = 0
    
    # Spatial correlation analysis
    if lidar_detection_count > 0 and radar_detection_count > 0:
        lidar_centroid = np.mean(lidar_in_bbox, axis=0)
        radar_centroid = np.mean(radar_in_bbox, axis=0)
        inter_sensor_distance = np.linalg.norm(lidar_centroid - radar_centroid)
        
        # Calculate spatial spread for each sensor
        lidar_spatial_spread = (np.std(lidar_in_bbox, axis=0) if lidar_detection_count > 1 else np.zeros(3))
        radar_spatial_spread = (np.std(radar_in_bbox, axis=0) if radar_detection_count > 1 else np.zeros(3))
        
        spatial_correlation_coefficient = 1 / (1 + inter_sensor_distance)
        dimensional_complementarity = np.mean(np.abs(lidar_spatial_spread - radar_spatial_spread))
    else:
        inter_sensor_distance = 10.0  # Penalty for single-sensor detection
        spatial_correlation_coefficient = 0.0
        dimensional_complementarity = 0.0
    
    # Fusion confidence metric
    if lidar_detection_count > 0 and radar_detection_count > 0:
        fusion_confidence_score = detection_balance_score * spatial_correlation_coefficient
    else:
        fusion_confidence_score = 0.1  # Low confidence for single-sensor cases
    
    return np.array([lidar_contribution_ratio, radar_contribution_ratio, detection_balance_score, inter_sensor_distance, spatial_correlation_coefficient, dimensional_complementarity, fusion_confidence_score])


def extract_features_for_sample(sample_data, truckscenes_instance, sensor_modality):
    """
    Extract features for all annotations in a single sample.
    
    Args:
        sample_data (dict): Sample data from TruckScenes
        truckscenes_instance: TruckScenes dataset instance
        sensor_modality (str): Type of sensor analysis ("LIDAR", "RADAR", "LIDAR+RADAR")
        
    Returns:
        tuple: (feature_matrix, label_vector) as numpy arrays
    """
    feature_list = []
    label_list = []
    
    # Load sensor data based on modality
    if sensor_modality == "LIDAR":
        sensor_channels = [key for key in sample_data['data'].keys() if key.startswith('LIDAR')]
        lidar_point_clouds = []
        radar_point_clouds = np.empty((0, 3))
        
        for channel in sensor_channels:
            data_token = sample_data['data'].get(channel)
            if data_token:
                sensor_data = truckscenes_instance.get('sample_data', data_token)
                file_path = os.path.join(DATASET_ROOT_PATH, sensor_data['filename'])
                if os.path.exists(file_path):
                    points = load_point_cloud_data(file_path)
                    if points.size > 0:
                        lidar_point_clouds.append(points)
        
        combined_point_cloud = (np.vstack(lidar_point_clouds) if lidar_point_clouds else np.empty((0, 3)))
        
    elif sensor_modality == "RADAR":
        sensor_channels = [key for key in sample_data['data'].keys() if key.startswith('RADAR')]
        radar_point_clouds = []
        lidar_point_clouds = np.empty((0, 3))
        
        for channel in sensor_channels:
            data_token = sample_data['data'].get(channel)
            if data_token:
                sensor_data = truckscenes_instance.get('sample_data', data_token)
                file_path = os.path.join(DATASET_ROOT_PATH, sensor_data['filename'])
                if os.path.exists(file_path):
                    points = load_point_cloud_data(file_path)
                    if points.size > 0:
                        radar_point_clouds.append(points)
        
        combined_point_cloud = (np.vstack(radar_point_clouds) if radar_point_clouds else np.empty((0, 3)))
        
    else:  # LIDAR+RADAR case
        lidar_channels = [key for key in sample_data['data'].keys() if key.startswith('LIDAR')]
        radar_channels = [key for key in sample_data['data'].keys() if key.startswith('RADAR')]
        
        # Load LIDAR data
        lidar_point_clouds = []
        for channel in lidar_channels:
            data_token = sample_data['data'].get(channel)
            if data_token:
                sensor_data = truckscenes_instance.get('sample_data', data_token)
                file_path = os.path.join(DATASET_ROOT_PATH, sensor_data['filename'])
                if os.path.exists(file_path):
                    points = load_point_cloud_data(file_path)
                    if points.size > 0:
                        lidar_point_clouds.append(points)
        
        # Load RADAR data
        radar_point_clouds = []
        for channel in radar_channels:
            data_token = sample_data['data'].get(channel)
            if data_token:
                sensor_data = truckscenes_instance.get('sample_data', data_token)
                file_path = os.path.join(DATASET_ROOT_PATH, sensor_data['filename'])
                if os.path.exists(file_path):
                    points = load_point_cloud_data(file_path)
                    if points.size > 0:
                        radar_point_clouds.append(points)
        
        lidar_point_clouds = (np.vstack(lidar_point_clouds) if lidar_point_clouds else np.empty((0, 3)))
        radar_point_clouds = (np.vstack(radar_point_clouds) if radar_point_clouds else np.empty((0, 3)))
        combined_point_cloud = (np.vstack([lidar_point_clouds, radar_point_clouds]) if (lidar_point_clouds.size > 0 or radar_point_clouds.size > 0) else np.empty((0, 3)))
    
    # Process each object annotation in the sample
    for annotation_token in sample_data['anns']:
        annotation_data = truckscenes_instance.get('sample_annotation', annotation_token)
        object_class_label = annotation_data.get('category_name', 'unknown')
        
        # Extract geometric features (common to all sensor modalities)
        geometric_features = extract_geometric_features(annotation_data)
        
        # Define bounding box for point filtering
        object_center = np.array(annotation_data['translation'])
        object_size = np.array(annotation_data['size'])
        
        # Normalize coordinates to vehicle frame
        normalized_center = np.array([
            (object_center[0] - 571700) / 100,
            (object_center[1] - 5367800) / 100,
            object_center[2] - 594
        ])
        
        bounding_box_min = normalized_center - object_size / 2
        bounding_box_max = normalized_center + object_size / 2
        
        # Filter points within object bounding box
        if combined_point_cloud.size > 0:
            points_in_object = combined_point_cloud[np.all((combined_point_cloud >= bounding_box_min) & (combined_point_cloud <= bounding_box_max), axis=1)]
        else:
            points_in_object = np.empty((0, 3))
        
        if sensor_modality == "LIDAR+RADAR":
            # Extract separate features for LIDAR and RADAR
            lidar_filtered = (lidar_point_clouds[np.all((lidar_point_clouds >= bounding_box_min) & (lidar_point_clouds <= bounding_box_max), axis=1)] if lidar_point_clouds.size > 0 else np.empty((0, 3)))
            
            radar_filtered = (radar_point_clouds[np.all((radar_point_clouds >= bounding_box_min) & (radar_point_clouds <= bounding_box_max), axis=1)] if radar_point_clouds.size > 0 else np.empty((0, 3)))
            
            lidar_features = extract_sensor_specific_features(lidar_filtered, annotation_data, "LIDAR")
            radar_features = extract_sensor_specific_features(radar_filtered, annotation_data, "RADAR")
            fusion_features = extract_fusion_sensor_features(lidar_point_clouds, radar_point_clouds, annotation_data)
            
            # Combine all feature types for fusion
            complete_feature_vector = np.concatenate([
                geometric_features,    # 9 geometric features
                lidar_features,        # 6 LIDAR-specific features
                radar_features,        # 5 RADAR-specific features
                fusion_features        # 7 fusion-specific features
            ])  # Total: 27 features for fusion
            
        else:
            # Extract sensor-specific features for single-sensor analysis
            sensor_features = extract_sensor_specific_features(points_in_object, annotation_data, sensor_modality)
            complete_feature_vector = np.concatenate([geometric_features, sensor_features])
        
        # Add regularization noise to improve generalization
        complete_feature_vector = add_gaussian_noise(complete_feature_vector, GAUSSIAN_NOISE_LEVEL)
        
        feature_list.append(complete_feature_vector)
        label_list.append(object_class_label)
    
    return np.array(feature_list), np.array(label_list)


def calculate_multiclass_specificity(true_labels, predicted_labels, class_labels):
    """
    Calculate specificity for multi-class classification.
    
    Args:
        true_labels (numpy.ndarray): True class labels
        predicted_labels (numpy.ndarray): Predicted class labels
        class_labels (list): List of unique class labels
        
    Returns:
        float: Average specificity across all classes
    """
    confusion_matrix_data = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    specificity_scores = []
    
    for class_index in range(len(class_labels)):
        true_positives = confusion_matrix_data[class_index, class_index]
        false_negatives = np.sum(confusion_matrix_data[class_index, :]) - true_positives
        false_positives = np.sum(confusion_matrix_data[:, class_index]) - true_positives
        true_negatives = np.sum(confusion_matrix_data) - true_positives - false_negatives - false_positives
        
        if true_negatives + false_positives > 0:
            class_specificity = true_negatives / (true_negatives + false_positives)
        else:
            class_specificity = 1.0
        
        specificity_scores.append(class_specificity)
    
    return np.mean(specificity_scores)


def map_class_names_to_readable(class_labels):
    """
    Map TruckScenes class names to readable format.
    
    Args:
        class_labels (list): List of original class names from TruckScenes
        
    Returns:
        list: List of readable class names
    """
    class_mapping = {
        'human.pedestrian.adult': 'Adult',
        'human.pedestrian.child': 'Child',
        'human.pedestrian.construction_worker': 'Worker',
        'human.pedestrian.personal_mobility': 'Personal Mobility',
        'human.pedestrian.police_officer': 'Police Officer',
        'human.pedestrian.stroller': 'Stroller',
        'human.pedestrian.wheelchair': 'Wheelchair',
        'animal': 'Animal',
        'vehicle.bicycle': 'Bicycle',
        'vehicle.bus.bendy': 'Bendy Bus',
        'vehicle.bus.rigid': 'Bus',
        'vehicle.car': 'Car',
        'vehicle.construction': 'Construction',
        'vehicle.emergency.ambulance': 'Ambulance',
        'vehicle.emergency.police': 'Police',
        'vehicle.motorcycle': 'Motorcycle',
        'vehicle.trailer': 'Trailer',
        'vehicle.truck': 'Truck',
        'vehicle.ego_trailer': 'Ego Vehicle trailer',
        'vehicle.other': 'Other vehicle',
        'vehicle.train': 'Train',
        'movable_object.barrier': 'Barrier',
        'movable_object.debris': 'Debris',
        'movable_object.pushable_pullable': 'Pushable/Pullable',
        'movable_object.trafficcone': 'Traffic cone',
        'static_object.bicycle_rack': 'Bicycle rack',
        'static_object.traffic_sign': 'Traffic sign',
        'flat.driveable_surface': 'Road',
        'flat.other': 'Other surface',
        'flat.sidewalk': 'Sidewalk',
        'flat.terrain': 'Terrain',
        'static.manmade': 'Manmade',
        'static.other': 'Other static',
        'static.vegetation': 'Vegetation',
        'vehicle.ego': 'Ego Vehicle'
    }
    
    return [class_mapping.get(class_name, class_name.replace('.', '_').title()) for class_name in class_labels]


def plot_confusion_matrix(true_labels, predicted_labels, class_labels, sensor_modality):
    """
    Generate and display confusion matrix for a sensor modality with improved readability.
    
    Args:
        true_labels (numpy.ndarray): True class labels
        predicted_labels (numpy.ndarray): Predicted class labels  
        class_labels (list): List of unique class labels
        sensor_modality (str): Sensor type for plot title
    """
    # Map class names to readable format
    readable_class_labels = map_class_names_to_readable(class_labels)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    
    # Calculate normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Determine figure size based on number of classes for better readability
    num_classes = len(class_labels)
    fig_width = max(18, num_classes * 1.5)
    fig_height = max(8, num_classes * 0.8)
    
    # Create figure with subplots for both absolute and normalized matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Configure font sizes based on matrix size
    if num_classes <= 5:
        annot_fontsize = 14
        title_fontsize = 16
        label_fontsize = 12
    elif num_classes <= 10:
        annot_fontsize = 10
        title_fontsize = 14
        label_fontsize = 10
    else:
        annot_fontsize = 8
        title_fontsize = 12
        label_fontsize = 8
    
    # Plot absolute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=readable_class_labels, yticklabels=readable_class_labels, ax=ax1, annot_kws={'size': annot_fontsize}, cbar_kws={'shrink': 0.8})
    ax1.set_title(f'{sensor_modality} - Absolute Confusion Matrix', fontsize=title_fontsize, pad=20)
    ax1.set_xlabel('Predicted Labels', fontsize=label_fontsize)
    ax1.set_ylabel('True Labels', fontsize=label_fontsize)
    
    # Rotate labels
    ax1.tick_params(axis='x', rotation=45, labelsize=label_fontsize-1)
    ax1.tick_params(axis='y', rotation=0, labelsize=label_fontsize-1)
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=readable_class_labels, yticklabels=readable_class_labels, 
                ax=ax2, annot_kws={'size': annot_fontsize},
                cbar_kws={'shrink': 0.8})
    ax2.set_title(f'{sensor_modality} - Normalized Confusion Matrix (%)', fontsize=title_fontsize, pad=20)
    ax2.set_xlabel('Predicted Labels', fontsize=label_fontsize)
    ax2.set_ylabel('True Labels', fontsize=label_fontsize)
    
    # Rotate labels
    ax2.tick_params(axis='x', rotation=45, labelsize=label_fontsize-1)
    ax2.tick_params(axis='y', rotation=0, labelsize=label_fontsize-1)
    
    # Adjust layout to prevent label cutoff with robust error handling
    try:
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.90)
    except (ValueError, UserWarning, RuntimeWarning) as e:
        print(f"ℹ️  Layout optimization: Using manual adjustment for confusion matrix")
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.88, hspace=0.3, wspace=0.3)
    except Exception as e:
        print(f"ℹ️  Layout optimization: Applied fallback layout settings for confusion matrix")
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.85)
    
    # Save the plot
    output_filename = f'{sensor_modality.replace("+", "_")}_confusion_matrix.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrix saved as: {output_filename}")
    
    plt.close()


def plot_roc_curves(true_labels, predicted_probabilities, class_labels, sensor_modality):
    """
    Generate and save ROC curves for multi-class classification with AUC scores.
    
    Args:
        true_labels (numpy.ndarray): True class labels
        predicted_probabilities (numpy.ndarray): Predicted class probabilities
        class_labels (list): List of unique class labels
        sensor_modality (str): Sensor type for plot title
    """
    # Map class names to readable format
    readable_class_labels = map_class_names_to_readable(class_labels)
    
    # Binarize the true labels for multi-class ROC
    binary_labels = label_binarize(true_labels, classes=class_labels)
    n_classes = len(class_labels)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], predicted_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(binary_labels.ravel(), predicted_probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Determine figure size and layout based on number of classes
    if n_classes <= 5:
        fig_width, fig_height = 12, 8
        legend_fontsize = 10
        title_fontsize = 14
        label_fontsize = 12
    elif n_classes <= 10:
        fig_width, fig_height = 14, 10
        legend_fontsize = 8
        title_fontsize = 12
        label_fontsize = 10
    else:
        fig_width, fig_height = 16, 12
        legend_fontsize = 6
        title_fontsize = 11
        label_fontsize = 9
    
    # Create the plot
    plt.figure(figsize=(fig_width, fig_height))
    
    # Plot ROC curves for each class
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{readable_class_labels[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=3, label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=3, label=f'Macro-avg (AUC = {roc_auc["macro"]:.3f})')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier')
    
    # Configure plot appearance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=label_fontsize)
    plt.ylabel('True Positive Rate', fontsize=label_fontsize)
    plt.title(f'{sensor_modality} - ROC Curves (Multi-class)', fontsize=title_fontsize, pad=20)
    
    # Add legend
    if n_classes <= 8:
        plt.legend(loc="lower right", fontsize=legend_fontsize)
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_fontsize)
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Adjust layout with robust error handling for tight_layout warnings
    try:
        plt.tight_layout()
    except (ValueError, UserWarning, RuntimeWarning) as e:
        if n_classes <= 8:
            plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.15)
        else:
            plt.subplots_adjust(left=0.1, right=0.80, top=0.90, bottom=0.15)
    except Exception as e:
        plt.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.15)
    
    # Save the plot
    output_filename = f'{sensor_modality.replace("+", "_")}_roc_curves.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"ROC curves saved as: {output_filename}")
    
    plt.close()


def plot_performance_scatter(evaluation_results):
    """
    Generate scatter plots showing performance metrics across folds for each sensor modality.
    
    Args:
        evaluation_results (dict): Dictionary containing performance metrics for each sensor modality
    """
    # Define metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy', 'specificity']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy', 'Specificity']
    
    # Prepare data for plotting
    sensor_modalities = list(evaluation_results.keys())
    n_folds = len(next(iter(evaluation_results.values()))['accuracy'])  # Get number of folds
    
    # Create figure with optimized size and layout for compact visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Color scheme for different sensor modalities
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[metric_idx]
        
        # Plot data for each sensor modality
        for sensor_idx, sensor in enumerate(sensor_modalities):
            metric_values = evaluation_results[sensor][metric]
            fold_numbers = np.arange(1, len(metric_values) + 1)
            
            # Scatter plot for individual folds
            ax.scatter(fold_numbers, metric_values, 
                      color=colors[sensor_idx % len(colors)], 
                      marker=markers[sensor_idx % len(markers)],
                      s=80, alpha=0.8, label=sensor, edgecolors='black', linewidth=0.5)
            
            # Connect points with a trend line
            ax.plot(fold_numbers, metric_values, 
                   color=colors[sensor_idx % len(colors)], 
                   alpha=0.5, linewidth=1.2, linestyle='--')
            
            # Add mean line
            mean_value = np.mean(metric_values)
            ax.axhline(y=mean_value, color=colors[sensor_idx % len(colors)], 
                      linestyle='-', alpha=0.8, linewidth=1.8)
        
        # Customize subplot with compact styling
        ax.set_xlabel('CV Fold', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xticks(np.arange(1, n_folds + 1))
        
        # Set y-axis limits for better visualization
        all_values = np.concatenate([evaluation_results[sensor][metric] for sensor in sensor_modalities])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.03 * y_range, y_max + 0.05 * y_range)
        
        # Add compact legend only to the upper right subplot
        if metric_idx == 2:  # Upper right position
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=False)
    
    # Title
    plt.suptitle('Performance Metrics Across Cross-Validation Folds',fontsize=14, fontweight='bold', y=0.96)
    
    # Apply optimized layout with minimal spacing
    try:
        plt.tight_layout(pad=1.5, h_pad=1.8, w_pad=1.5)
        plt.subplots_adjust(top=0.90, right=0.88)
    except (ValueError, UserWarning, RuntimeWarning, Exception):
        plt.subplots_adjust(left=0.08, right=0.86, top=0.90, bottom=0.12, hspace=0.30, wspace=0.25)
    
    # Save the plot
    output_filename = 'performance_variability_scatter_plots.png'

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2, format='png')
    except Exception as e:
        print(f"Saving with alternative method due to layout optimization")
        plt.savefig(output_filename, dpi=300, facecolor='white', format='png')
    
    print(f"Performance scatter plots saved as: {output_filename}")

    # Close the plot
    plt.close()


def evaluate_sensor_classification_performance(truckscenes_instance, sensor_modality):
    """
    Evaluate classification performance for a specific sensor modality using k-fold cross-validation.
    
    Args:
        truckscenes_instance: TruckScenes dataset instance
        sensor_modality (str): Sensor type to evaluate ("LIDAR", "RADAR", "LIDAR+RADAR")
        
    Returns:
        dict: Dictionary containing performance metrics arrays
    """
    
    # Collect features and labels from all samples
    all_feature_matrices = []
    all_label_vectors = []
    
    for sample_data in tqdm(truckscenes_instance.sample, desc=f"Extracting {sensor_modality} features", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        
        feature_matrix, label_vector = extract_features_for_sample(sample_data, truckscenes_instance, sensor_modality)
        
        if len(feature_matrix) > 0:
            all_feature_matrices.append(feature_matrix)
            all_label_vectors.append(label_vector)

    if not all_feature_matrices:
        print(f"No valid data found for {sensor_modality}")
        return None
    
    # Combine all features and labels
    feature_dataset = np.vstack(all_feature_matrices)
    label_dataset = np.hstack(all_label_vectors)
        
    # Filter classes with sufficient samples for reliable cross-validation
    unique_classes, class_counts = np.unique(label_dataset, return_counts=True)
    valid_classes = unique_classes[class_counts >= 5]  # Minimum 5 samples per class
    
    if len(valid_classes) < 2:
        print(f"Insufficient classes with adequate samples for {sensor_modality}")
        return None
    
    # Filter dataset to include only valid classes
    valid_sample_mask = np.isin(label_dataset, valid_classes)
    feature_dataset = feature_dataset[valid_sample_mask]
    label_dataset = label_dataset[valid_sample_mask]
        
    # Standardize features for improved classifier performance
    feature_scaler = StandardScaler()
    standardized_features = feature_scaler.fit_transform(feature_dataset)
    
    # Initialize k-fold cross-validation
    cross_validator = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Initialize performance metrics storage
    performance_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'balanced_accuracy': [],
        'specificity': []
    }
    
    # Storage for confusion matrix data
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    
    # Perform k-fold cross-validation
    for fold_index, (training_indices, testing_indices) in enumerate(cross_validator.split(standardized_features, label_dataset)):
        
        # Split data into training and testing sets
        training_features = standardized_features[training_indices]
        testing_features = standardized_features[testing_indices]
        training_labels = label_dataset[training_indices]
        testing_labels = label_dataset[testing_indices]
        
        # Configure classifier based on sensor modality
        if sensor_modality == "LIDAR+RADAR":
            # Enhanced parameters for fusion to handle increased feature complexity
            classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=RANDOM_SEED + fold_index
            )
        else:
            # Standard parameters for single-sensor analysis
            classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_SEED + fold_index
            )
        
        # Train classifier and make predictions
        classifier.fit(training_features, training_labels)
        predicted_labels = classifier.predict(testing_features)
        predicted_probabilities = classifier.predict_proba(testing_features)
        
        # Calculate performance metrics
        accuracy = accuracy_score(testing_labels, predicted_labels)
        precision = precision_score(testing_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(testing_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(testing_labels, predicted_labels, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(testing_labels, predicted_labels)
        specificity = calculate_multiclass_specificity(testing_labels, predicted_labels, valid_classes)
        
        # Store metrics
        performance_metrics['accuracy'].append(accuracy)
        performance_metrics['precision'].append(precision)
        performance_metrics['recall'].append(recall)
        performance_metrics['f1_score'].append(f1)
        performance_metrics['balanced_accuracy'].append(balanced_acc)
        performance_metrics['specificity'].append(specificity)
        
        # Collect predictions for confusion matrix and ROC curves
        all_true_labels.extend(testing_labels)
        all_predicted_labels.extend(predicted_labels)
        all_predicted_probabilities.append(predicted_probabilities)
    
    # Generate confusion matrix
    plot_confusion_matrix(np.array(all_true_labels), np.array(all_predicted_labels), valid_classes, sensor_modality)
    
    # Generate ROC curves
    all_probabilities_combined = np.vstack(all_predicted_probabilities)
    plot_roc_curves(np.array(all_true_labels), all_probabilities_combined, 
                   valid_classes, sensor_modality)
    
    print()
            
    # Convert lists to numpy arrays for statistical analysis
    return {metric: np.array(values) for metric, values in performance_metrics.items()}


def visualize_sensor_comparison(truckscenes_instance):
    """
    Generate a comparison visualization of LIDAR, RADAR, and LIDAR+RADAR sensors for the first sample.

    Args:
        truckscenes_instance: TruckScenes dataset instance
    """
    first_scene = truckscenes_instance.scene[0]
    first_sample_token = first_scene['first_sample_token']
    sample_data = truckscenes_instance.get('sample', first_sample_token)
    
    # Collect sensor data
    sensor_data = {}
    for sensor_type in ["LIDAR", "RADAR"]:
        channels = [key for key in sample_data['data'].keys() if key.startswith(sensor_type)]
        points = []
        for channel in channels:
            data_token = sample_data['data'].get(channel)
            if data_token:
                sensor_info = truckscenes_instance.get('sample_data', data_token)
                file_path = os.path.join(DATASET_ROOT_PATH, sensor_info['filename'])
                if os.path.exists(file_path):
                    pc_data = load_point_cloud_data(file_path)
                    if pc_data.size > 0:
                        points.append(pc_data)
        sensor_data[sensor_type] = np.vstack(points) if points else np.empty((0, 3))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["LIDAR Only", "RADAR Only", "LIDAR + RADAR Fusion"]
    data_sets = [sensor_data["LIDAR"], sensor_data["RADAR"], np.vstack([sensor_data["LIDAR"], sensor_data["RADAR"]]) if sensor_data["LIDAR"].size > 0 and sensor_data["RADAR"].size > 0 else np.empty((0, 3))]
    colors = ['blue', 'red', 'purple']
    
    for idx, (ax, title, data, color) in enumerate(zip(axes, titles, data_sets, colors)):
        if data.size > 0:
            scatter = ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis', s=0.5, alpha=0.6)
            plt.colorbar(scatter, ax=ax, shrink=0.8, label='Height (m)')
            ax.text(0.02, 0.98, f'Points: {len(data):,}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, fontsize=12, ha='center', va='center')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('sensor_comparison_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
    print()
    print(f"Sensor comparison visualization saved as: sensor_comparison_visualization.png")
    plt.close()


def main():
    # Initialize TruckScenes dataset
    truckscenes_dataset = TruckScenes(DATASET_VERSION, DATASET_ROOT_PATH, verbose=True)

    # Define sensor modalities for evaluation
    sensor_modalities = ["LIDAR", "RADAR", "LIDAR+RADAR"]
    evaluation_results = {}
    
    # Evaluate each sensor modality
    for modality in sensor_modalities:

        performance_metrics = evaluate_sensor_classification_performance(truckscenes_dataset, modality)
        if performance_metrics is not None:
            evaluation_results[modality] = performance_metrics
    
    # Generate performance summary table
    if evaluation_results:
        print("\nClassification performance results:\n")
        
        # Prepare data for tabular presentation
        summary_data = []
        for modality, metrics in evaluation_results.items():
            summary_row = {
                'Sensor': modality,
                'Accuracy': f"{np.mean(metrics['accuracy']):.3f} ± {np.std(metrics['accuracy']):.3f}",
                'Precision': f"{np.mean(metrics['precision']):.3f} ± {np.std(metrics['precision']):.3f}",
                'Recall': f"{np.mean(metrics['recall']):.3f} ± {np.std(metrics['recall']):.3f}",
                'F1 Score': f"{np.mean(metrics['f1_score']):.3f} ± {np.std(metrics['f1_score']):.3f}",
                'Balanced Accuracy': f"{np.mean(metrics['balanced_accuracy']):.3f} ± {np.std(metrics['balanced_accuracy']):.3f}",
                'Specificity': f"{np.mean(metrics['specificity']):.3f} ± {np.std(metrics['specificity']):.3f}"
            }
            summary_data.append(summary_row)
        
        # Display results table
        results_dataframe = pd.DataFrame(summary_data)
        print(results_dataframe.to_string(index=False))
        
        # Generate sensor comparison visualization
        visualize_sensor_comparison(truckscenes_dataset)
        
        # Generate performance scatter plots
        plot_performance_scatter(evaluation_results)
    
    else:
        print("No evaluation results obtained for any sensor modality")


if __name__ == "__main__":
    main()
