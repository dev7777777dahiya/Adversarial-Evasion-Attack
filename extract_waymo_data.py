import os
import cv2
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# === Config ===
GCS_PATHS = [
    'segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord',
    'segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord',
    'segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
]

EXPORT_FRAMES_DIR = "exported_frames"
EXPORT_LIDAR_DIR = "exported_lidar"
MAX_FRAMES = 596

# === Setup ===
os.makedirs(EXPORT_FRAMES_DIR, exist_ok=True)
os.makedirs(EXPORT_LIDAR_DIR, exist_ok=True)

def extract_waymo_data(gcs_paths, max_frames):
    count = 0
    for gcs_path in gcs_paths:
        print(f"Processing: {gcs_path}")
        dataset = tf.data.TFRecordDataset(gcs_path, compression_type='')
        for record in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(record.numpy())

            # ==== Extract front camera frame ====
            front_img_saved = False
            for image in frame.images:
                if image.name == open_dataset.CameraName.FRONT:
                    img = tf.image.decode_jpeg(image.image).numpy()
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    frame_path = os.path.join(EXPORT_FRAMES_DIR, f"frame_{count:05d}.jpg")
                    cv2.imwrite(frame_path, img_bgr)
                    print(f"Saved frame: {frame_path}")
                    front_img_saved = True
                    break

            # ==== Extract LiDAR ====
            try:
                # Parse range images and camera projections (unpack 4 values)
                range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)

                # Convert range images to point cloud
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose
                )

                all_points = np.concatenate(points, axis=0)
                lidar_path = os.path.join(EXPORT_LIDAR_DIR, f"lidar_{count:05d}.npy")
                np.save(lidar_path, all_points[:, :3])  # Save x, y, z
                print(f"Saved lidar: {lidar_path}")
            except Exception as e:
                print(f"LiDAR extraction failed at frame {count}: {e}")
                continue

            if front_img_saved:
                count += 1
            if count >= max_frames:
                print(f"Reached frame limit ({max_frames}).")
                return

if __name__ == "__main__":
    extract_waymo_data(GCS_PATHS, MAX_FRAMES)
