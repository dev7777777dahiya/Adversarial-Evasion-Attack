# Adversarial Defense and Temporal Pattern Learning for Autonomous Vehicle Perception

## Description

This project, developed by **Dev Dahiya**, proposes a defense framework to enhance the perception robustness of autonomous vehicles (AVs) against adversarial attacks. The system combines **input transformation (WebP compression)**, **deep learning-based object detection (YOLOv8)**, **multi-modal data fusion (LiDAR + Camera)**, and **temporal consistency enforcement using an LSTM network**.

Built on the **Waymo Open Dataset**, the solution targets real-world adversarial challenges by providing resilience against perturbations that mislead perception modules of AV systems. The approach effectively handles both digital and potential physical perturbations in the environment while maintaining real-time operational feasibility.

---

## Concept and Motivation

Autonomous vehicles rely on machine learning models for interpreting sensor data. These models are highly susceptible to adversarial attacks—small, carefully crafted perturbations that cause misclassification or object misdetection. Given that real-time, multi-sensor perception is critical for safe AV operation, this project aims to:

1. **Defend Against Image Perturbations:** WebP compression is used to remove high-frequency adversarial noise.
2. **Integrate Multi-Modal Sensing:** LiDAR’s spatial consistency is fused with camera data to reduce dependency on a single modality.
3. **Maintain Temporal Consistency:** A Bidirectional LSTM processes sequences of frames, learning object movement and behavior across time to predict and correct frame-level detection failures.
4. **Ensure Real-Time Feasibility:** Lightweight architectures like YOLOv8-nano and minimal overhead preprocessing ensure deployment potential in real AV systems.

---

## System Architecture

1. **Data Extraction:**

   * Utilizes Waymo Open Dataset TFRecord files.
   * Extracts synchronized **front-camera frames (JPEG)** and **3D LiDAR point clouds (NumPy)** ensuring millisecond-level temporal alignment.

2. **Adversarial Defense:**

   * Each frame is WebP compressed and decompressed with a quality factor of 75, mitigating adversarial perturbations without significant image degradation.

3. **Object Detection (YOLOv8):**

   * Compressed frames are passed through **YOLOv8-nano** for real-time object detection.
   * Outputs include bounding boxes, class labels, confidence scores.

4. **LiDAR Fusion:**

   * Mean (x, y, z) values of LiDAR points are extracted per frame.
   * YOLO outputs are combined with these LiDAR features into a unified feature vector.

5. **Temporal Consistency Learning (LSTM):**

   * A Bidirectional LSTM processes sequences (length=10) of the combined feature vectors.
   * Predicts object classes in subsequent frames, improving detection reliability when YOLO misclassifies due to adversarial perturbations.

6. **Visualization and Output:**

   * Detections and LSTM predictions are annotated on frames.
   * Results can be saved as a video for review.

---

## Installation

```bash
git clone https://github.com/dev7777777dahiya/Autonomous-Vehicle-Perception.git
cd Autonomous-Vehicle-Perception
pip install -r requirements.txt
```

---

## Usage

1. **Extract Waymo Data:**

   Update TFRecord paths in `extract_waymo_data.py`:

   ```python
   GCS_PATHS = ['file1.tfrecord', 'file2.tfrecord', 'file3.tfrecord']
   ```

   Run:

   ```bash
   python extract_waymo_data.py
   ```

2. **Run Detection and Temporal Analysis:**

   Configure YOLO & LSTM in `webp_yolo_lstm.py`:

   ```python
   TRAIN_LSTM = True  # to train LSTM
   # or
   TRAIN_LSTM = False # to use existing weights
   ```

   Execute:

   ```bash
   python webp_yolo_lstm.py
   ```

---

## Explanation of Logic and Flow

1. **Data Flow:**

   * Extract synchronized **camera frames** and **LiDAR data** from Waymo Dataset.
   * Each camera frame undergoes **WebP compression** to mitigate adversarial noise.
   * YOLOv8 detects objects and outputs class IDs, bounding boxes, and confidence scores.
   * **LiDAR point cloud means (x, y, z)** are calculated and appended to the YOLO feature vector.
   * The **feature vector (YOLO + LiDAR)** is fed to the **LSTM** for temporal pattern recognition.

2. **LSTM Training:**

   * Trained to predict the object class of the (t+1) frame using the past 10 frames.
   * Helps detect and recover from YOLO misclassifications caused by adversarial noise.

3. **Inference:**

   * During runtime, LSTM uses the history buffer (deque of past 10 feature vectors) to predict the object class for the current frame.
   * Frame is annotated with YOLO and LSTM labels for comparison.
   * Optionally saved as an output video.

---

## Experimental Results Summary

| Configuration              | Detection Accuracy | False Negatives | LSTM Recovery |
| -------------------------- | ------------------ | --------------- | ------------- |
| YOLOv8 (Clean Data)        | \~88%              | Low             | -             |
| YOLOv8 (Adversarial Input) | \~42%              | High            | 0             |
| YOLOv8 + WebP + LSTM       | \~73%              | Moderate        | 31%           |

* **WebP Preprocessing** reduced noise impact by \~15-20%.
* **LSTM** recovered detections missed by YOLO in 31% of cases, using temporal context.
* **LiDAR Fusion** provided stability against visual perturbations.

---

## Future Enhancements

* **Adversarial Training:** Incorporate adversarial examples in LSTM training.
* **Uncertainty Awareness:** Bayesian LSTM for predictive uncertainty estimation.
* **Geometry-Aware Fusion:** Transition to early-fusion models.
* **Real-Time Deployment:** Optimize models for embedded AV hardware.

---

## Contributing

Contributions are welcome. Fork the repository and submit pull requests with proposed changes.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Special thanks to **Waymo Open Dataset** and **Ultralytics YOLO** team for providing open-source tools essential for this work.
