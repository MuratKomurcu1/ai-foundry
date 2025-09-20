import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import time
from dataclasses import dataclass
import logging

# YOLO için
try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Detectron2 için
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

@dataclass
class Detection:
    """Tespit sonucu"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int

@dataclass
class DetectionResult:
    """Tespit analiz sonucu"""
    image_path: str
    detections: List[Detection]
    processing_time: float
    model_used: str
    image_size: Tuple[int, int]
    
class YOLODetector:
    """YOLO tabanlı nesne tespiti"""
    
    def __init__(self, model_size: str = "yolov8n"):
        """
        Args:
            model_size: Model boyutu (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO library not installed")
        
        self.model_size = model_size
        self.model = YOLO(f"{model_size}.pt")
        self.class_names = self.model.names
        
        logging.info(f"YOLO model loaded: {model_size}")
    
    def detect_image(self, image_path: str, conf_threshold: float = 0.5) -> DetectionResult:
        """Görüntüde nesne tespiti"""
        start_time = time.time()
        
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # YOLO ile tespit
        results = self.model(image_path, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Bbox koordinatları
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Center ve area hesapla
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        area=area
                    ))
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            image_path=image_path,
            detections=detections,
            processing_time=processing_time,
            model_used=f"YOLO-{self.model_size}",
            image_size=(w, h)
        )
    
    def detect_video(self, video_path: str, output_path: str = None, 
                    conf_threshold: float = 0.5, show_fps: bool = True):
        """Video'da nesne tespiti"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            # Video writer setup
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # YOLO ile tespit
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Sonuçları çiz
            annotated_frame = results[0].plot()
            
            processing_time = time.time() - start_time
            total_time += processing_time
            frame_count += 1
            
            # FPS bilgisi ekle
            if show_fps:
                fps_text = f"FPS: {1/processing_time:.1f}"
                cv2.putText(annotated_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Kaydet veya göster
            if output_path:
                out.write(annotated_frame)
            else:
                cv2.imshow('YOLO Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        avg_fps = frame_count / total_time if total_time > 0 else 0
        logging.info(f"Video processing completed. Average FPS: {avg_fps:.2f}")
    
    def detect_webcam(self, conf_threshold: float = 0.5):
        """Webcam'den real-time tespit"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # YOLO ile tespit
            results = self.model(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # FPS hesapla ve göster
            fps = 1 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('YOLO Webcam', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

class Detectron2Detector:
    """Detectron2 tabanlı nesne tespiti"""
    
    def __init__(self, model_name: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"):
        """
        Args:
            model_name: Detectron2 model adı
        """
        if not DETECTRON2_AVAILABLE:
            raise ImportError("Detectron2 library not installed")
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        
        logging.info(f"Detectron2 model loaded: {model_name}")
    
    def detect_image(self, image_path: str) -> DetectionResult:
        """Görüntüde nesne tespiti"""
        start_time = time.time()
        
        # Görüntüyü yükle
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Tespit
        outputs = self.predictor(image)
        
        detections = []
        instances = outputs["instances"]
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.metadata.thing_classes[cls]
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                
                detections.append(Detection(
                    class_name=class_name,
                    confidence=float(score),
                    bbox=(x1, y1, x2, y2),
                    center=(center_x, center_y),
                    area=area
                ))
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            image_path=image_path,
            detections=detections,
            processing_time=processing_time,
            model_used="Detectron2",
            image_size=(w, h)
        )

class FaceDetector:
    """Yüz tespiti ve tanıma"""
    
    def __init__(self):
        # OpenCV DNN yüz dedektörü
        self.face_net = cv2.dnn.readNetFromTensorflow(
            'opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt'
        )
        
        # Yaş ve cinsiyet tahmin modelleri
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
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """Yüz tespiti ve analizi"""
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Blob oluştur
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Yüz ROI
                face_roi = image[y1:y2, x1:x2]
                
                # Yaş ve cinsiyet tahmini
                age, gender = self._predict_age_gender(face_roi)
                
                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(confidence),
                    'age': age,
                    'gender': gender
                })
        
        return faces
    
    def _predict_age_gender(self, face_roi):
        """Yaş ve cinsiyet tahmini"""
        try:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            # Cinsiyet tahmini
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Yaş tahmini
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            
            return age, gender
            
        except Exception as e:
            logging.warning(f"Age/Gender prediction failed: {e}")
            return "Unknown", "Unknown"

class CustomObjectDetector:
    """Custom eğitilmiş modeller için wrapper"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Args:
            model_path: Model dosya yolu
            config_path: Konfigürasyon dosya yolu
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # PyTorch modeli yükle
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()
        
        logging.info(f"Custom model loaded: {model_path}")
    
    def detect(self, image_path: str) -> List[Detection]:
        """Custom model ile tespit"""
        # Bu kısım model tipine göre customize edilecek
        # Örnek implementation
        
        image = cv2.imread(image_path)
        # Model-specific preprocessing
        # ...
        
        # Inference
        # results = self.model(preprocessed_image)
        
        # Post-processing
        # ...
        
        return []  # Placeholder

class ObjectTracker:
    """Nesne takibi (tracking)"""
    
    def __init__(self, tracker_type: str = "CSRT"):
        """
        Args:
            tracker_type: Tracker tipi (CSRT, KCF, MOSSE, etc.)
        """
        self.tracker_type = tracker_type
        self.trackers = []
        
        # Tracker factory
        self.tracker_types = {
            'BOOSTING': cv2.legacy.TrackerBoosting_create,
            'MIL': cv2.legacy.TrackerMIL_create,
            'KCF': cv2.legacy.TrackerKCF_create,
            'TLD': cv2.legacy.TrackerTLD_create,
            'MEDIANFLOW': cv2.legacy.TrackerMedianFlow_create,
            'MOSSE': cv2.legacy.TrackerMOSSE_create,
            'CSRT': cv2.TrackerCSRT_create
        }
    
    def initialize_tracker(self, frame, bbox):
        """Tracker'ı başlat"""
        tracker = self.tracker_types[self.tracker_type]()
        tracker.init(frame, bbox)
        self.trackers.append(tracker)
        return len(self.trackers) - 1  # Tracker ID
    
    def update_trackers(self, frame):
        """Tüm tracker'ları güncelle"""
        results = []
        
        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)
            results.append({
                'tracker_id': i,
                'success': success,
                'bbox': bbox if success else None
            })
        
        return results

class DetectionAnalyzer:
    """Tespit sonuçlarını analiz etme"""
    
    @staticmethod
    def analyze_detections(results: List[DetectionResult]) -> Dict:
        """Toplu tespit analizi"""
        if not results:
            return {}
        
        all_detections = []
        for result in results:
            all_detections.extend(result.detections)
        
        # Sınıf istatistikleri
        class_counts = {}
        confidence_scores = []
        areas = []
        
        for det in all_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            confidence_scores.append(det.confidence)
            areas.append(det.area)
        
        # İstatistikler
        analysis = {
            'total_detections': len(all_detections),
            'unique_classes': len(class_counts),
            'class_distribution': class_counts,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'avg_area': np.mean(areas) if areas else 0,
            'avg_processing_time': np.mean([r.processing_time for r in results])
        }
        
        return analysis
    
    @staticmethod
    def visualize_results(image_path: str, result: DetectionResult, 
                         save_path: str = None, show: bool = True):
        """Tespit sonuçlarını görselleştir"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        
        # Her tespit için bbox çiz
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            
            # Rectangle çiz
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Label ekle
            label = f"{det.class_name}: {det.confidence:.2f}"
            plt.text(x1, y1-5, label, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.title(f"Detection Results - {result.model_used}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()

# Demo fonksiyonu
def demo_object_detection():
    """Nesne tespiti demo"""
    print("🎯 Gelişmiş Nesne Tespit Sistemi Demo")
    print("=" * 50)
    
    # YOLO demo
    if YOLO_AVAILABLE:
        print("🤖 YOLO Detector Test")
        try:
            detector = YOLODetector("yolov8n")
            print(f"✅ Model yüklendi: {detector.model_size}")
            print(f"📋 Sınıf sayısı: {len(detector.class_names)}")
            print("💡 Gerçek test için bir görüntü yolu verin:")
            print("   result = detector.detect_image('path/to/image.jpg')")
        except Exception as e:
            print(f"❌ YOLO yüklenemedi: {e}")
    
    # Detectron2 demo
    if DETECTRON2_AVAILABLE:
        print("\n🔬 Detectron2 Test")
        try:
            detector = Detectron2Detector()
            print("✅ Detectron2 modeli yüklendi")
        except Exception as e:
            print(f"❌ Detectron2 yüklenemedi: {e}")
    
    # Face detector demo
    print("\n👤 Face Detector Test")
    try:
        face_detector = FaceDetector()
        print("✅ Face detector hazır")
        print("💡 Yüz tespiti için:")
        print("   faces = face_detector.detect_faces('path/to/face_image.jpg')")
    except Exception as e:
        print(f"❌ Face detector yüklenemedi: {e}")
    
    # Örnek kullanım
    print("\n📝 Örnek Kullanım Kodları:")
    print("""
# YOLO ile nesne tespiti
detector = YOLODetector('yolov8n')
result = detector.detect_image('image.jpg')
print(f"Tespit edilen nesne sayısı: {len(result.detections)}")

# Real-time webcam tespiti
detector.detect_webcam()

# Video'da nesne tespiti
detector.detect_video('input.mp4', 'output.mp4')

# Sonuçları görselleştir
DetectionAnalyzer.visualize_results('image.jpg', result)
    """)

if __name__ == "__main__":
    demo_object_detection()