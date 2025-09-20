import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
from torchvision import transforms
from PIL import Image
import face_recognition
import dlib

# Emotion recognition
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False

@dataclass
class FaceDetection:
    """Yüz tespit sonucu"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: Optional[np.ndarray] = None
    encoding: Optional[np.ndarray] = None

@dataclass
class FaceAnalysis:
    """Yüz analiz sonucu"""
    face_id: Optional[str]
    detection: FaceDetection
    emotion: Optional[Dict[str, float]] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    similarity_score: Optional[float] = None

class FaceDetector:
    """Yüz tespit sınıfı"""
    
    def __init__(self, method: str = "haar"):
        """
        Args:
            method: Tespit yöntemi ("haar", "dnn", "mtcnn", "dlib")
        """
        self.method = method
        self.setup_detector()
    
    def setup_detector(self):
        """Detektörü kurma"""
        if self.method == "haar":
            # OpenCV Haar Cascade
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        elif self.method == "dnn":
            # OpenCV DNN
            self.net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
        
        elif self.method == "dlib":
            # Dlib HOG detector
            self.detector = dlib.get_frontal_face_detector()
            
        elif self.method == "mtcnn":
            # MTCNN (requires mtcnn package)
            try:
                from mtcnn import MTCNN
                self.detector = MTCNN()
            except ImportError:
                logging.warning("MTCNN not available, falling back to Haar")
                self.method = "haar"
                self.setup_detector()
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[FaceDetection]:
        """Yüz tespiti"""
        faces = []
        
        if self.method == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in detections:
                faces.append(FaceDetection(
                    bbox=(x, y, w, h),
                    confidence=1.0  # Haar doesn't provide confidence
                ))
        
        elif self.method == "dnn":
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.net.setInput(blob)
            detections = self.net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    faces.append(FaceDetection(
                        bbox=(x1, y1, x2-x1, y2-y1),
                        confidence=float(confidence)
                    ))
        
        elif self.method == "dlib":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector(gray)
            
            for detection in detections:
                x = detection.left()
                y = detection.top()
                w = detection.width()
                h = detection.height()
                
                faces.append(FaceDetection(
                    bbox=(x, y, w, h),
                    confidence=1.0  # Dlib doesn't provide confidence
                ))
        
        elif self.method == "mtcnn":
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_image)
            
            for detection in detections:
                if detection['confidence'] > confidence_threshold:
                    x, y, w, h = detection['box']
                    keypoints = detection['keypoints']
                    
                    # Landmarks array oluştur
                    landmarks = np.array([
                        keypoints['left_eye'],
                        keypoints['right_eye'],
                        keypoints['nose'],
                        keypoints['mouth_left'],
                        keypoints['mouth_right']
                    ])
                    
                    faces.append(FaceDetection(
                        bbox=(x, y, w, h),
                        confidence=detection['confidence'],
                        landmarks=landmarks
                    ))
        
        return faces

class FacialLandmarkDetector:
    """Yüz landmark tespiti"""
    
    def __init__(self):
        # Dlib 68-point landmark predictor
        try:
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            self.detector = dlib.get_frontal_face_detector()
            self.available = True
        except Exception as e:
            logging.warning(f"Landmark detector not available: {e}")
            self.available = False
    
    def detect_landmarks(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """68-point landmark tespiti"""
        if not self.available:
            return None
        
        x, y, w, h = face_bbox
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Dlib rectangle oluştur
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Landmarks tespit et
        landmarks = self.predictor(gray, rect)
        
        # Points array'e çevir
        points = np.zeros((68, 2), dtype=int)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)
        
        return points
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Landmarks çizme"""
        if landmarks is None:
            return image
        
        output = image.copy()
        
        # Landmark grupları
        landmark_groups = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'mouth': list(range(48, 68))
        }
        
        colors = {
            'jaw': (255, 0, 0),
            'right_eyebrow': (0, 255, 0),
            'left_eyebrow': (0, 255, 0),
            'nose': (0, 0, 255),
            'right_eye': (255, 255, 0),
            'left_eye': (255, 255, 0),
            'mouth': (255, 0, 255)
        }
        
        # Her grup için çiz
        for group, indices in landmark_groups.items():
            color = colors[group]
            for i in indices:
                if i < len(landmarks):
                    cv2.circle(output, tuple(landmarks[i]), 2, color, -1)
        
        return output

class EmotionRecognizer:
    """Duygu tanıma"""
    
    def __init__(self):
        self.setup_emotion_model()
    
    def setup_emotion_model(self):
        """Emotion model kurulumu"""
        if FER_AVAILABLE:
            self.fer = FER(mtcnn=True)
            self.available = True
            logging.info("✅ FER emotion model loaded")
        else:
            # Basit emotion classifier (placeholder)
            self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.available = False
            logging.warning("⚠️ FER not available, using placeholder")
    
    def predict_emotion(self, face_image: np.ndarray) -> Dict[str, float]:
        """Emotion prediction"""
        if self.available:
            try:
                # FER ile emotion detection
                result = self.fer.detect_emotions(face_image)
                
                if result:
                    emotions = result[0]['emotions']
                    return emotions
                else:
                    return self._get_neutral_emotions()
            except Exception as e:
                logging.error(f"Emotion prediction error: {e}")
                return self._get_neutral_emotions()
        else:
            # Placeholder random emotions
            import random
            emotions = {}
            total = 1.0
            
            for emotion in self.emotions[:-1]:
                score = random.uniform(0, total/len(self.emotions))
                emotions[emotion] = score
                total -= score
            
            emotions[self.emotions[-1]] = max(0, total)
            return emotions
    
    def _get_neutral_emotions(self) -> Dict[str, float]:
        """Neutral emotion durumu"""
        return {
            'angry': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
            'happy': 0.1,
            'sad': 0.0,
            'surprise': 0.0,
            'neutral': 0.9
        }

class AgeGenderEstimator:
    """Yaş ve cinsiyet tahmini"""
    
    def __init__(self):
        self.setup_models()
    
    def setup_models(self):
        """Yaş ve cinsiyet modellerini kurma"""
        try:
            # OpenCV DNN modelleri
            self.age_net = cv2.dnn.readNetFromCaffe(
                'age_deploy.prototxt',
                'age_net.caffemodel'
            )
            
            self.gender_net = cv2.dnn.readNetFromCaffe(
                'gender_deploy.prototxt',
                'gender_net.caffemodel'
            )
            
            self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                           '(38-43)', '(48-53)', '(60-100)']
            self.gender_list = ['Male', 'Female']
            
            self.available = True
            logging.info("✅ Age/Gender models loaded")
            
        except Exception as e:
            logging.warning(f"⚠️ Age/Gender models not available: {e}")
            self.available = False
    
    def predict_age_gender(self, face_image: np.ndarray) -> Tuple[str, str]:
        """Yaş ve cinsiyet tahmini"""
        if not self.available:
            return "Unknown", "Unknown"
        
        try:
            # Preprocess
            blob = cv2.dnn.blobFromImage(
                face_image, 1.0, (227, 227), 
                (78.4263377603, 87.7689143744, 114.895847746), 
                swapRB=False
            )
            
            # Gender prediction
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Age prediction
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            
            return age, gender
            
        except Exception as e:
            logging.error(f"Age/Gender prediction error: {e}")
            return "Unknown", "Unknown"

class FaceRecognitionSystem:
    """Ana yüz tanıma sistemi"""
    
    def __init__(self, db_path: str = "face_database.db"):
        self.db_path = db_path
        
        # Components
        self.face_detector = FaceDetector("dnn")
        self.landmark_detector = FacialLandmarkDetector()
        self.emotion_recognizer = EmotionRecognizer()
        self.age_gender_estimator = AgeGenderEstimator()
        
        # Database
        self.setup_database()
        
        # Known faces
        self.known_faces = {}
        self.load_known_faces()
    
    def setup_database(self):
        """SQLite veritabanı kurulumu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP,
                seen_count INTEGER DEFAULT 0
            )
        ''')
        
        # Face logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                emotion TEXT,
                age TEXT,
                gender TEXT,
                FOREIGN KEY (face_id) REFERENCES faces (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_known_faces(self):
        """Bilinen yüzleri yükleme"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, encoding FROM faces")
        for name, encoding_blob in cursor.fetchall():
            encoding = pickle.loads(encoding_blob)
            self.known_faces[name] = encoding
        
        conn.close()
        logging.info(f"✅ {len(self.known_faces)} known faces loaded")
    
    def add_face(self, name: str, image: np.ndarray) -> bool:
        """Yeni yüz ekleme"""
        # Yüz tespiti
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            logging.error("No face detected in image")
            return False
        
        if len(faces) > 1:
            logging.warning("Multiple faces detected, using the first one")
        
        # En büyük yüzü al
        largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        x, y, w, h = largest_face.bbox
        
        # Face crop
        face_image = image[y:y+h, x:x+w]
        
        # Face encoding
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        
        if not encodings:
            logging.error("Could not generate face encoding")
            return False
        
        encoding = encodings[0]
        
        # Database'e kaydet
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO faces (name, encoding, last_seen, seen_count)
                VALUES (?, ?, ?, 1)
            ''', (name, pickle.dumps(encoding), datetime.now()))
            
            conn.commit()
            self.known_faces[name] = encoding
            
            logging.info(f"✅ Face added: {name}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding face: {e}")
            return False
        finally:
            conn.close()
    
    def recognize_face(self, face_encoding: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        """Yüz tanıma"""
        if not self.known_faces:
            return "Unknown", 0.0
        
        # Tüm bilinen yüzlerle karşılaştır
        known_names = list(self.known_faces.keys())
        known_encodings = list(self.known_faces.values())
        
        # Mesafe hesapla
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        # En yakın eşleşme
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]
        
        if best_distance < threshold:
            name = known_names[best_match_index]
            confidence = 1 - best_distance
            
            # Database güncelle
            self.update_face_seen(name)
            
            return name, confidence
        else:
            return "Unknown", 0.0
    
    def update_face_seen(self, name: str):
        """Yüz görülme sayısını güncelleme"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE faces 
            SET last_seen = ?, seen_count = seen_count + 1
            WHERE name = ?
        ''', (datetime.now(), name))
        
        conn.commit()
        conn.close()
    
    def analyze_image(self, image: np.ndarray) -> List[FaceAnalysis]:
        """Görüntü analizi"""
        results = []
        
        # Yüz tespiti
        faces = self.face_detector.detect_faces(image)
        
        for face in faces:
            x, y, w, h = face.bbox
            
            # Face crop
            face_image = image[y:y+h, x:x+w]
            
            # Face encoding
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            
            face_analysis = FaceAnalysis(
                face_id=None,
                detection=face,
                emotion=None,
                age=None,
                gender=None,
                similarity_score=None
            )
            
            if encodings:
                encoding = encodings[0]
                face.encoding = encoding
                
                # Face recognition
                name, confidence = self.recognize_face(encoding)
                face_analysis.face_id = name
                face_analysis.similarity_score = confidence
            
            # Emotion recognition
            if face_image.size > 0:
                emotions = self.emotion_recognizer.predict_emotion(face_image)
                face_analysis.emotion = emotions
                
                # Age/Gender estimation
                age, gender = self.age_gender_estimator.predict_age_gender(face_image)
                face_analysis.age = age
                face_analysis.gender = gender
            
            # Landmarks
            if self.landmark_detector.available:
                landmarks = self.landmark_detector.detect_landmarks(image, face.bbox)
                face.landmarks = landmarks
            
            results.append(face_analysis)
        
        return results
    
    def process_video(self, video_source: int = 0, save_path: str = None):
        """Video işleme (webcam veya dosya)"""
        cap = cv2.VideoCapture(video_source)
        
        if save_path:
            # Video writer setup
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analiz
            analyses = self.analyze_image(frame)
            
            # Çizim
            output_frame = self.draw_analysis(frame, analyses)
            
            # Kaydet veya göster
            if save_path:
                out.write(output_frame)
            else:
                cv2.imshow('Face Recognition', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if save_path:
            out.release()
        cv2.destroyAllWindows()
    
    def draw_analysis(self, image: np.ndarray, analyses: List[FaceAnalysis]) -> np.ndarray:
        """Analiz sonuçlarını çizme"""
        output = image.copy()
        
        for analysis in analyses:
            face = analysis.detection
            x, y, w, h = face.bbox
            
            # Face rectangle
            color = (0, 255, 0) if analysis.face_id != "Unknown" else (0, 0, 255)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Face ID
            label = f"{analysis.face_id}"
            if analysis.similarity_score:
                label += f" ({analysis.similarity_score:.2f})"
            
            cv2.putText(output, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Age/Gender
            if analysis.age and analysis.gender:
                demo_label = f"{analysis.age}, {analysis.gender}"
                cv2.putText(output, demo_label, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Top emotion
            if analysis.emotion:
                top_emotion = max(analysis.emotion.items(), key=lambda x: x[1])
                emotion_label = f"{top_emotion[0]}: {top_emotion[1]:.2f}"
                cv2.putText(output, emotion_label, (x, y + h + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Landmarks
            if face.landmarks is not None:
                output = self.landmark_detector.draw_landmarks(output, face.landmarks)
        
        return output
    
    def get_statistics(self) -> Dict:
        """İstatistikler"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total faces
        cursor.execute("SELECT COUNT(*) FROM faces")
        total_faces = cursor.fetchone()[0]
        
        # Most seen faces
        cursor.execute("""
            SELECT name, seen_count, last_seen 
            FROM faces 
            ORDER BY seen_count DESC 
            LIMIT 5
        """)
        top_faces = cursor.fetchall()
        
        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) 
            FROM face_logs 
            WHERE timestamp > datetime('now', '-1 hour')
        """)
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_faces': total_faces,
            'known_faces': len(self.known_faces),
            'top_faces': top_faces,
            'recent_activity': recent_activity
        }

# Demo ve test fonksiyonları
def demo_face_recognition():
    """Face recognition demo"""
    print("👤 Face Recognition System Demo")
    print("=" * 50)
    
    # Sistem oluştur
    face_system = FaceRecognitionSystem()
    
    print(f"🧠 Components loaded:")
    print(f"   Face Detector: {face_system.face_detector.method}")
    print(f"   Landmarks: {'✅' if face_system.landmark_detector.available else '❌'}")
    print(f"   Emotions: {'✅' if face_system.emotion_recognizer.available else '❌'}")
    print(f"   Age/Gender: {'✅' if face_system.age_gender_estimator.available else '❌'}")
    
    # Statistics
    stats = face_system.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Known faces: {stats['known_faces']}")
    print(f"   Total faces in DB: {stats['total_faces']}")
    print(f"   Recent activity: {stats['recent_activity']}")
    
    print(f"\n💡 Usage Examples:")
    print(f"""
# Yeni yüz ekleme
face_system.add_face("John Doe", image)

# Webcam'den real-time tanıma
face_system.process_video(0)

# Görüntü analizi
analyses = face_system.analyze_image(image)

# Video işleme
face_system.process_video("input.mp4", "output.mp4")
    """)
    
    print("✨ Demo completed!")

if __name__ == "__main__":
    demo_face_recognition()