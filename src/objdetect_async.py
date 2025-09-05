import asyncio
import concurrent.futures
import numpy as np
import cv2 as cv
import mediapipe as mp
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AsyncPullUpDetector:
    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.5):
        """Initialize async pull-up detector with MediaPipe"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Thread pool for CPU-intensive operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # State variables for tracking pull-up phases
        self.reset_state()
    
    def reset_state(self):
        """Reset pull-up tracking state"""
        self.successful_count = 0
        self.start_reached = False
        self.pullup_reached = False
        self.top_reached = False
        self.descent_reached = False
        self.initial_shoulder_position = None
        
        # For debugging/logging
        self.start_message_shown = False
        self.pullup_message_shown = False
        self.top_message_shown = False
        self.descent_message_shown = False
    
    async def process_video_async(self, video_path: str, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asynchronously process video and return landmark sequences and labels
        """
        try:
            # Process video in thread pool to avoid blocking
            landmark_sequences, label_sequences = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._process_video_sync,
                video_path,
                sequence_length
            )
            
            return landmark_sequences, label_sequences
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise
    
    def _process_video_sync(self, video_path: str, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchronous video processing (runs in thread pool)
        """
        cap = cv.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Initialize MediaPipe pose
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Storage for sequences
        landmark_sequences = []
        label_sequences = []
        frame_sequence = []
        label_sequence = []
        
        frame_count = 0
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.info(f"Processed {frame_count} frames from video")
                    break
                
                frame_count += 1
                
                # Process frame
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Extract landmarks
                    landmarks = self._extract_landmarks(results, frame.shape)
                    
                    if landmarks:
                        # Get features and phase label
                        features = self._landmarks_to_features(landmarks)
                        current_phase = self._analyze_pullup_phase(landmarks)
                        
                        frame_sequence.append(features)
                        label_sequence.append(current_phase)
                        
                        # Create sequence when we have enough frames
                        if len(frame_sequence) == sequence_length:
                            landmark_sequences.append(np.array(frame_sequence.copy()))
                            label_sequences.append(np.array(label_sequence.copy()))
                            
                            # Sliding window: remove oldest frame
                            frame_sequence.pop(0)
                            label_sequence.pop(0)
                
                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    logger.info(f"Processing progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
            pose.close()
        
        return np.array(landmark_sequences), np.array(label_sequences)
    
    def _extract_landmarks(self, results, frame_shape: Tuple[int, int, int]) -> Dict[str, Tuple[float, float]]:
        """Extract relevant landmarks from MediaPipe results"""
        if not results.pose_landmarks:
            return {}
        
        h, w = frame_shape[:2]
        landmarks = {}
        
        try:
            # Extract key landmarks
            pose_landmarks = results.pose_landmarks.landmark
            
            landmarks['wrist'] = (
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * h
            )
            landmarks['elbow'] = (
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y * h
            )
            landmarks['shoulder'] = (
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
            )
            landmarks['hip'] = (
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h
            )
            
        except (AttributeError, IndexError) as e:
            logger.warning(f"Error extracting landmarks: {e}")
            return {}
        
        return landmarks
    
    def _landmarks_to_features(self, landmarks: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Convert landmarks to feature vector for RNN input"""
        # Flatten landmarks to feature vector
        features = []
        for landmark_name in ['wrist', 'elbow', 'shoulder', 'hip']:
            if landmark_name in landmarks:
                features.extend(landmarks[landmark_name])
            else:
                features.extend([0.0, 0.0])  # Default values for missing landmarks
        
        return np.array(features, dtype=np.float32)
    
    def _analyze_pullup_phase(self, landmarks: Dict[str, Tuple[float, float]]) -> int:
        """
        Analyze current pull-up phase based on landmarks
        Returns: 0=start, 1=pulling, 2=top, 3=descent
        """
        if not landmarks or 'wrist' not in landmarks:
            return 0
        
        try:
            # Extract positions and calculate angles
            wrist_y = landmarks['wrist'][1]
            elbow_angle = self._calculate_angle(
                landmarks['elbow'], landmarks['shoulder'], landmarks['wrist']
            )
            shoulder_angle = self._calculate_angle(
                landmarks['elbow'], landmarks['shoulder'], landmarks['hip']
            )
            shoulder_y = landmarks['shoulder'][1]
            
            # Define thresholds
            threshold_bent_angle = 70
            current_phase = 0
            
            # Phase detection logic (adapted from original)
            if not self.pullup_reached:
                if not self.start_reached:
                    self.initial_shoulder_position = shoulder_y
                self.start_reached = True
                current_phase = 0
            
            if shoulder_angle <= 160:
                self.pullup_reached = True
                current_phase = 1
            
            if (self.pullup_reached and not self.descent_reached and 
                wrist_y < shoulder_y and elbow_angle < threshold_bent_angle):
                self.top_reached = True
                current_phase = 2
            
            if self.top_reached and elbow_angle < 5:
                self.descent_reached = True
                current_phase = 3
            
            # Check for completed pull-up
            if (self.start_reached and self.pullup_reached and 
                self.top_reached and self.descent_reached):
                if elbow_angle > 120:
                    self.successful_count += 1
                    logger.info(f'Pull-up #{self.successful_count} completed!')
                    self._reset_pullup_state()
            
            return current_phase
            
        except Exception as e:
            logger.warning(f"Error analyzing pull-up phase: {e}")
            return 0
    
    def _reset_pullup_state(self):
        """Reset state after completing a pull-up"""
        self.start_reached = False
        self.pullup_reached = False
        self.top_reached = False
        self.descent_reached = False
        self.start_message_shown = False
        self.pullup_message_shown = False
        self.top_message_shown = False
        self.descent_message_shown = False
    
    @staticmethod
    def _calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            ab = a - b
            bc = c - b
            
            # Handle zero vectors
            norm_ab = np.linalg.norm(ab)
            norm_bc = np.linalg.norm(bc)
            
            if norm_ab == 0 or norm_bc == 0:
                return 0.0
            
            cosine_angle = np.dot(ab, bc) / (norm_ab * norm_bc)
            
            # Clamp to valid range for arccos
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            angle = np.arccos(cosine_angle)
            return float(np.degrees(angle))
            
        except Exception:
            return 0.0
    
    async def process_frame_async(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process single frame asynchronously
        Returns: (features, phase_label)
        """
        try:
            # Process frame in thread pool
            features, phase = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._process_frame_sync,
                frame
            )
            
            return features, phase
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return np.zeros(8, dtype=np.float32), 0
    
    def _process_frame_sync(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Synchronous frame processing"""
        pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        try:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = self._extract_landmarks(results, frame.shape)
                if landmarks:
                    features = self._landmarks_to_features(landmarks)
                    phase = self._analyze_pullup_phase(landmarks)
                    return features, phase
            
            return np.zeros(8, dtype=np.float32), 0
            
        finally:
            pose.close()
    
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)