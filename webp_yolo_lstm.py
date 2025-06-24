import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from collections import deque
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

FRAME_DIR = "exported_frames0.07"
LIDAR_DIR = "exported_lidar"
SEQUENCE_LENGTH = 10
SAVE_VIDEO = True
OUTPUT_VIDEO = "output_yolo_lstm007.mp4"
TRAIN_LSTM = False

LSTM_WEIGHTS_PATH = "lstm_weights.pth"  # Path to save/load LSTM weights

def webp_compress_decompress(img, quality=75):
    """Compress and decompress an image using WebP to remove adversarial perturbations."""
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    retval, buf = cv2.imencode('.webp', img, encode_param)
    if not retval:
        raise ValueError("Could not encode image to WebP format.")
    img_webp = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img_webp

class TemporalConsistencyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class YOLOWithLSTM:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.classes = self.model.names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = len(self.classes) + 5 + 3  # Additional 3 for LiDAR mean x, y, z
        hidden_size = 128
        num_layers = 1
        self.lstm = TemporalConsistencyLSTM(input_size, hidden_size, num_layers, len(self.classes)).to(self.device)
        self.history = deque(maxlen=SEQUENCE_LENGTH)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.input_size = input_size

        # === Load LSTM weights if present ===
        if os.path.exists(LSTM_WEIGHTS_PATH) and not TRAIN_LSTM:
            self.lstm.load_state_dict(torch.load(LSTM_WEIGHTS_PATH, map_location=self.device))
            self.lstm.eval()
            print(f"Loaded LSTM weights from {LSTM_WEIGHTS_PATH}")
        else:
            print("No LSTM weights found or TRAIN_LSTM=True. Will train from scratch.")

    def load_lidar(self, index):
        lidar_file = os.path.join(LIDAR_DIR, f"lidar_{index:05d}.npy")
        if os.path.exists(lidar_file):
            points = np.load(lidar_file)
            if points.shape[0] > 0:
                return np.mean(points[:, :3], axis=0)  # mean x, y, z
        return np.zeros(3)

    def detect_frame(self, frame, index):
        frame = webp_compress_decompress(frame, quality=75)
        img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        results = self.model.predict(frame, conf=0.5)
        boxes = results[0].boxes
        if not boxes:
            return torch.zeros(self.input_size)
        best = max(boxes, key=lambda box: box.conf.item())
        class_id = int(best.cls.cpu().numpy()[0])
        confidence = float(best.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = best.xyxy.cpu().numpy()[0]
        vector = torch.zeros(self.input_size)
        vector[class_id] = 1.0
        vector[len(self.classes):len(self.classes) + 5] = torch.tensor([x1, y1, x2 - x1, y2 - y1, confidence])
        vector[len(self.classes) + 5:] = torch.tensor(self.load_lidar(index), dtype=torch.float32)
        assert vector.shape[0] == self.input_size, f"Feature vector size {vector.shape[0]} does not match expected {self.input_size}"
        return vector

    def train_lstm(self, num_epochs=5, batch_size=4):
        sequences = []
        labels = []
        buffer = deque(maxlen=SEQUENCE_LENGTH + 1)
        image_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])
        for i, file in enumerate(tqdm(image_files, desc="Processing frames", leave=False, ncols=100)):
            frame = cv2.imread(os.path.join(FRAME_DIR, file))
            feature = self.detect_frame(frame, i)
            buffer.append(feature)
            if len(buffer) == SEQUENCE_LENGTH + 1:
                sequences.append(torch.stack(list(buffer)[:-1]))
                target_vector = buffer[-1]
                class_id = torch.argmax(target_vector[:len(self.classes)])
                labels.append(class_id)
        if not sequences:
            print("No sequences found for LSTM training.")
            return
        dataset = TensorDataset(torch.stack(sequences), torch.tensor(labels))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.lstm.parameters(), lr=1e-3)
        self.lstm.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.lstm(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        # === Save weights after training ===
        torch.save(self.lstm.state_dict(), LSTM_WEIGHTS_PATH)
        print(f"Saved LSTM weights to {LSTM_WEIGHTS_PATH}")
        self.lstm.eval()

    def run(self):
        image_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])
        writer = None
        if SAVE_VIDEO and image_files:
            first_img = cv2.imread(os.path.join(FRAME_DIR, image_files[0]))
            height, width = first_img.shape[:2]
            writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
        for i, file in enumerate(tqdm(image_files, desc="Running YOLO+LSTM", leave=False, ncols=100)):
            img_path = os.path.join(FRAME_DIR, file)
            frame = cv2.imread(img_path)
            feature = self.detect_frame(frame, i)
            self.history.append(feature.to(self.device))
            label = "Unknown"
            if len(self.history) == SEQUENCE_LENGTH:
                sequence = torch.stack(list(self.history)).unsqueeze(0)
                with torch.no_grad():
                    output = self.lstm(sequence)
                    predicted_class = torch.argmax(output, dim=1).item()
                    label = self.classes[predicted_class]
            annotated = frame.copy()
            # Draw YOLO detections
            results = self.model.predict(frame, conf=0.5)
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                class_id = int(box.cls.cpu().numpy()[0])
                confidence = float(box.conf.cpu().numpy()[0])
                class_name = self.classes[class_id]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Draw LSTM label
            cv2.putText(annotated, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if SAVE_VIDEO and writer is not None:
                writer.write(annotated)
            #cv2.imshow("YOLO + LSTM", annotated)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        if writer:
            writer.release()
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = YOLOWithLSTM()
    if TRAIN_LSTM:
        detector.train_lstm(num_epochs=5)
    detector.run()
