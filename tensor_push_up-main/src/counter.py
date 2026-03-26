"""
Action Counter State Machine Module

This module implements state machines for counting push-ups and jumping jacks
with anti-jitter and cooldown mechanisms for stable counting.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PushUpState(Enum):
    """States for push-up action detection."""
    NEUTRAL = "neutral"      # Initial state, not in exercise
    HIGH = "high"             # Arms extended, body in upper position
    LOW = "low"               # Elbows bent, body in lower position
    TRANSITION_DOWN = "transition_down"  # Moving from high to low
    TRANSITION_UP = "transition_up"      # Moving from low to high


class JumpingJackState(Enum):
    """States for jumping jack action detection."""
    NEUTRAL = "neutral"      # Initial state
    CLOSED = "closed"        # Arms down, feet together
    OPEN = "open"            # Arms up, feet apart
    TRANSITION_OUT = "transition_out"    # Moving from closed to open
    TRANSITION_IN = "transition_in"      # Moving from open to closed


class PushUpCounter:
    """
    State machine for counting push-up repetitions.

    Based on the requirements:
    - Standard push-ups only (hands only, no kneeling variations)
    - Body must remain straight (no sagging)
    - Elbow angle <= 90° at bottom position
    - Complete cycle (HIGH -> LOW -> HIGH) counts as one rep

    Attributes:
        count: Total number of completed push-ups
        state: Current state of the state machine
        stability_buffer: Buffer for state transition stability
        cooldown_frames: Frames remaining in cooldown period
    """

    def __init__(
        self,
        high_angle_threshold: float = 150.0,
        low_angle_threshold: float = 90.0,
        stability_frames: int = 3,
        cooldown_frames: int = 10,
        torso_angle_threshold: float = 30.0
    ):
        """
        Initialize the Push-Up Counter.

        Args:
            high_angle_threshold: Elbow angle for high position (degrees)
            low_angle_threshold: Elbow angle for low position (degrees)
            stability_frames: Frames to maintain position before state transition
            cooldown_frames: Frames to wait after counting before next count
            torso_angle_threshold: Maximum torso deviation from horizontal (degrees)
        """
        self.count = 0
        self.state = PushUpState.NEUTRAL
        self.stability_buffer = []  # Track recent angle states
        self.cooldown_frames = cooldown_frames
        self.current_cooldown = 0
        self.transition_start_frame = 0
        self.last_count_frame = -100

        # Parameters
        self.high_angle_threshold = high_angle_threshold
        self.low_angle_threshold = low_angle_threshold
        self.stability_frames = stability_frames
        self.torso_angle_threshold = torso_angle_threshold

        # Statistics
        self.total_frames = 0
        self.in_exercise = False

        logger.info(f"PushUpCounter initialized: high_angle={high_angle_threshold}°, "
                   f"low_angle={low_angle_threshold}°, stability={stability_frames}")

    def process_frame(
        self,
        keypoints: Optional[np.ndarray],
        angles: Optional[Dict[str, float]],
        frame_idx: int = 0
    ) -> Dict[str, any]:
        """
        Process a single frame and update counter state.

        Args:
            keypoints: Pose keypoints array (33, 4)
            angles: Dictionary of joint angles
            frame_idx: Current frame index

        Returns:
            Dictionary with current state, count, and debug info
        """
        self.total_frames += 1

        # Decrement cooldown
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        # No valid pose detected
        if keypoints is None or angles is None:
            result = {
                'count': self.count,
                'state': self.state.value,
                'elbow_angle': None,
                'torso_straight': None,
                'in_cooldown': self.current_cooldown > 0,
                'transition': False
            }

            # Reset to neutral if no pose for extended period
            if self.state != PushUpState.NEUTRAL and not self.in_exercise:
                self.state = PushUpState.NEUTRAL
                self.stability_buffer.clear()

            return result

        # Extract key angles
        left_elbow = angles.get('left_elbow', 180)
        right_elbow = angles.get('right_elbow', 180)
        avg_elbow = (left_elbow + right_elbow) / 2

        # Check torso alignment (shoulder-hip angle relative to horizontal)
        torso_angle = self._calculate_torso_angle(keypoints)
        torso_straight = abs(torso_angle) < self.torso_angle_threshold

        # Update state machine
        transition = self._update_state(avg_elbow, torso_straight, frame_idx)

        result = {
            'count': self.count,
            'state': self.state.value,
            'elbow_angle': avg_elbow,
            'torso_angle': torso_angle,
            'torso_straight': torso_straight,
            'in_cooldown': self.current_cooldown > 0,
            'transition': transition
        }

        return result

    def _calculate_torso_angle(self, keypoints: np.ndarray) -> float:
        """
        Calculate torso angle relative to horizontal.

        Args:
            keypoints: Pose keypoints array

        Returns:
            Torso angle in degrees (0° is horizontal)
        """
        # Use shoulder-hip vector
        left_shoulder = keypoints[11, :2]
        right_shoulder = keypoints[12, :2]
        left_hip = keypoints[23, :2]
        right_hip = keypoints[24, :2]

        # Average shoulder and hip positions
        shoulder = (left_shoulder + right_shoulder) / 2
        hip = (left_hip + right_hip) / 2

        # Vector from hip to shoulder (body axis)
        body_vector = shoulder - hip

        # Calculate angle from horizontal
        angle = np.degrees(np.arctan2(body_vector[1], body_vector[0]))

        return angle

    def _update_state(
        self,
        elbow_angle: float,
        torso_straight: bool,
        frame_idx: int
    ) -> bool:
        """
        Update state machine based on current pose.

        Args:
            elbow_angle: Average elbow angle
            torso_straight: Whether torso is within threshold
            frame_idx: Current frame index

        Returns:
            True if a count was incremented
        """
        transition = False

        # Determine position based on elbow angle
        if elbow_angle >= self.high_angle_threshold:
            position = 'HIGH'
        elif elbow_angle <= self.low_angle_threshold:
            position = 'LOW'
        else:
            position = 'MIDDLE'

        # Add to stability buffer
        self.stability_buffer.append(position)
        if len(self.stability_buffer) > self.stability_frames:
            self.stability_buffer.pop(0)

        # Check if buffer is stable
        if len(self.stability_buffer) < self.stability_frames:
            return False

        is_stable = all(p == self.stability_buffer[0] for p in self.stability_buffer)

        if not is_stable:
            return False

        stable_position = self.stability_buffer[0]

        # State transitions
        if self.current_cooldown > 0:
            # In cooldown, only maintain current state
            pass
        elif not torso_straight and self.state not in [PushUpState.NEUTRAL]:
            # Invalid form, reset to NEUTRAL
            logger.warning(f"Torso not straight at frame {frame_idx}, resetting")
            self.state = PushUpState.NEUTRAL
            self.stability_buffer.clear()
        else:
            transition = self._transition_states(stable_position, frame_idx)

        return transition

    def _transition_states(self, position: str, frame_idx: int) -> bool:
        """
        Handle state transitions based on stable position.

        Args:
            position: Stable position (HIGH, LOW, or MIDDLE)
            frame_idx: Current frame index

        Returns:
            True if a count was incremented
        """
        if position == 'MIDDLE':
            # In transition, update state accordingly
            if self.state == PushUpState.HIGH:
                self.state = PushUpState.TRANSITION_DOWN
            elif self.state == PushUpState.LOW:
                self.state = PushUpState.TRANSITION_UP
            return False

        if position == 'HIGH':
            if self.state == PushUpState.NEUTRAL:
                # Detected start of push-up
                self.state = PushUpState.HIGH
                self.in_exercise = True
                logger.debug(f"Push-up started at frame {frame_idx}")
            elif self.state == PushUpState.LOW:
                # Completing a rep: LOW -> HIGH
                if frame_idx - self.last_count_frame > self.cooldown_frames:
                    self.count += 1
                    self.current_cooldown = self.cooldown_frames
                    self.last_count_frame = frame_idx
                    self.state = PushUpState.HIGH
                    logger.info(f"Push-up counted! Total: {self.count} at frame {frame_idx}")
                    return True
            elif self.state == PushUpState.TRANSITION_UP:
                # Transition complete
                self.state = PushUpState.HIGH
            elif self.state == PushUpState.TRANSITION_DOWN:
                # Reversed direction
                self.state = PushUpState.HIGH

        elif position == 'LOW':
            if self.state == PushUpState.HIGH or self.state == PushUpState.TRANSITION_DOWN:
                # Reached bottom position
                self.state = PushUpState.LOW

        return False

    def reset(self):
        """Reset the counter to initial state."""
        self.count = 0
        self.state = PushUpState.NEUTRAL
        self.stability_buffer.clear()
        self.current_cooldown = 0
        self.total_frames = 0
        self.in_exercise = False
        self.last_count_frame = -100
        logger.info("PushUpCounter reset")


class JumpingJackCounter:
    """
    State machine for counting jumping jack repetitions.

    Based on the requirements:
    - Arms straight overhead touching at top
    - Feet wider than shoulders at bottom
    - Arms and legs should be synchronized
    - 45° side view

    Attributes:
        count: Total number of completed jumping jacks
        state: Current state of the state machine
        stability_buffer: Buffer for state transition stability
        cooldown_frames: Frames remaining in cooldown period
    """

    def __init__(
        self,
        open_ankle_threshold: float = 0.3,
        closed_ankle_threshold: float = 0.1,
        wrist_shoulder_threshold: float = 0.05,
        stability_frames: int = 3,
        cooldown_frames: int = 10
    ):
        """
        Initialize the Jumping Jack Counter.

        Args:
            open_ankle_threshold: Normalized ankle distance for open position
            closed_ankle_threshold: Normalized ankle distance for closed position
            wrist_shoulder_threshold: Wrist Y position relative to shoulder
            stability_frames: Frames to maintain position before state transition
            cooldown_frames: Frames to wait after counting
        """
        self.count = 0
        self.state = JumpingJackState.NEUTRAL
        self.stability_buffer = []
        self.current_cooldown = 0
        self.last_count_frame = -100

        # Parameters
        self.open_ankle_threshold = open_ankle_threshold
        self.closed_ankle_threshold = closed_ankle_threshold
        self.wrist_shoulder_threshold = wrist_shoulder_threshold
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames

        # Statistics
        self.total_frames = 0
        self.in_exercise = False

        logger.info(f"JumpingJackCounter initialized: open_ankle={open_ankle_threshold}, "
                   f"closed_ankle={closed_ankle_threshold}, stability={stability_frames}")

    def process_frame(
        self,
        keypoints: Optional[np.ndarray],
        angles: Optional[Dict[str, float]],
        frame_idx: int = 0
    ) -> Dict[str, any]:
        """
        Process a single frame and update counter state.

        Args:
            keypoints: Pose keypoints array (33, 4)
            angles: Dictionary of joint angles (not used but kept for interface consistency)
            frame_idx: Current frame index

        Returns:
            Dictionary with current state, count, and debug info
        """
        self.total_frames += 1

        # Decrement cooldown
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

        # No valid pose detected
        if keypoints is None:
            result = {
                'count': self.count,
                'state': self.state.value,
                'ankle_distance': None,
                'arms_up': None,
                'in_cooldown': self.current_cooldown > 0,
                'transition': False
            }

            if self.state != JumpingJackState.NEUTRAL and not self.in_exercise:
                self.state = JumpingJackState.NEUTRAL
                self.stability_buffer.clear()

            return result

        # Extract features
        ankle_distance = self._calculate_ankle_distance(keypoints)
        arms_up = self._check_arms_up(keypoints)

        # Update state machine
        transition = self._update_state(ankle_distance, arms_up, frame_idx)

        result = {
            'count': self.count,
            'state': self.state.value,
            'ankle_distance': ankle_distance,
            'arms_up': arms_up,
            'in_cooldown': self.current_cooldown > 0,
            'transition': transition
        }

        return result

    def _calculate_ankle_distance(self, keypoints: np.ndarray) -> float:
        """
        Calculate normalized distance between ankles.

        Args:
            keypoints: Pose keypoints array

        Returns:
            Normalized ankle distance
        """
        left_ankle = keypoints[27, :2]
        right_ankle = keypoints[28, :2]

        # Calculate distance
        distance = np.linalg.norm(left_ankle - right_ankle)

        # Normalize by shoulder width
        left_shoulder = keypoints[11, :2]
        right_shoulder = keypoints[12, :2]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_width > 0:
            normalized_distance = distance / shoulder_width
        else:
            normalized_distance = distance

        return normalized_distance

    def _check_arms_up(self, keypoints: np.ndarray) -> bool:
        """
        Check if arms are raised above shoulders.

        Args:
            keypoints: Pose keypoints array

        Returns:
            True if arms are above shoulders
        """
        left_wrist = keypoints[15, :2]
        right_wrist = keypoints[16, :2]
        left_shoulder = keypoints[11, :2]
        right_shoulder = keypoints[12, :2]

        # Check if wrists are above shoulders (smaller Y value)
        left_arm_up = left_wrist[1] < left_shoulder[1] - self.wrist_shoulder_threshold
        right_arm_up = right_wrist[1] < right_shoulder[1] - self.wrist_shoulder_threshold

        return left_arm_up and right_arm_up

    def _update_state(
        self,
        ankle_distance: float,
        arms_up: bool,
        frame_idx: int
    ) -> bool:
        """
        Update state machine based on current pose.

        Args:
            ankle_distance: Normalized distance between ankles
            arms_up: Whether arms are raised
            frame_idx: Current frame index

        Returns:
            True if a count was incremented
        """
        transition = False

        # Determine position
        if ankle_distance >= self.open_ankle_threshold and arms_up:
            position = 'OPEN'
        elif ankle_distance <= self.closed_ankle_threshold and not arms_up:
            position = 'CLOSED'
        else:
            position = 'TRANSITION'

        # Add to stability buffer
        self.stability_buffer.append(position)
        if len(self.stability_buffer) > self.stability_frames:
            self.stability_buffer.pop(0)

        # Check if buffer is stable
        if len(self.stability_buffer) < self.stability_frames:
            return False

        is_stable = all(p == self.stability_buffer[0] for p in self.stability_buffer)

        if not is_stable:
            return False

        stable_position = self.stability_buffer[0]

        # State transitions
        if self.current_cooldown > 0:
            pass
        else:
            transition = self._transition_states(stable_position, frame_idx)

        return transition

    def _transition_states(self, position: str, frame_idx: int) -> bool:
        """
        Handle state transitions based on stable position.

        Args:
            position: Stable position (OPEN, CLOSED, or TRANSITION)
            frame_idx: Current frame index

        Returns:
            True if a count was incremented
        """
        if position == 'TRANSITION':
            if self.state == JumpingJackState.CLOSED:
                self.state = JumpingJackState.TRANSITION_OUT
            elif self.state == JumpingJackState.OPEN:
                self.state = JumpingJackState.TRANSITION_IN
            return False

        if position == 'OPEN':
            if self.state == JumpingJackState.NEUTRAL:
                self.state = JumpingJackState.OPEN
                self.in_exercise = True
                logger.debug(f"Jumping jack started at frame {frame_idx}")
            elif self.state == JumpingJackState.CLOSED or self.state == JumpingJackState.TRANSITION_OUT:
                self.state = JumpingJackState.OPEN

        elif position == 'CLOSED':
            if self.state == JumpingJackState.OPEN or self.state == JumpingJackState.TRANSITION_IN:
                # Completing a rep: OPEN -> CLOSED
                if frame_idx - self.last_count_frame > self.cooldown_frames:
                    self.count += 1
                    self.current_cooldown = self.cooldown_frames
                    self.last_count_frame = frame_idx
                    self.state = JumpingJackState.CLOSED
                    logger.info(f"Jumping jack counted! Total: {self.count} at frame {frame_idx}")
                    return True
            elif self.state == JumpingJackState.TRANSITION_OUT:
                # Reversed direction
                self.state = JumpingJackState.CLOSED

        return False

    def reset(self):
        """Reset the counter to initial state."""
        self.count = 0
        self.state = JumpingJackState.NEUTRAL
        self.stability_buffer.clear()
        self.current_cooldown = 0
        self.total_frames = 0
        self.in_exercise = False
        self.last_count_frame = -100
        logger.info("JumpingJackCounter reset")


class ExerciseCounter:
    """
    Unified counter for multiple exercise types.

    This class manages multiple exercise counters and provides a unified
    interface for counting different exercise types.
    """

    def __init__(
        self,
        pushup_config: Optional[Dict] = None,
        jumping_jack_config: Optional[Dict] = None
    ):
        """
        Initialize the unified exercise counter.

        Args:
            pushup_config: Configuration dictionary for push-up counter
            jumping_jack_config: Configuration dictionary for jumping jack counter
        """
        pushup_config = pushup_config or {}
        jumping_jack_config = jumping_jack_config or {}

        self.pushup_counter = PushUpCounter(**pushup_config)
        self.jumping_jack_counter = JumpingJackCounter(**jumping_jack_config)

        logger.info("ExerciseCounter initialized with PushUp and JumpingJack")

    def process_frame(
        self,
        keypoints: Optional[np.ndarray],
        angles: Optional[Dict[str, float]],
        exercise_type: str = 'auto',
        frame_idx: int = 0
    ) -> Dict[str, any]:
        """
        Process a single frame for the specified or auto-detected exercise.

        Args:
            keypoints: Pose keypoints array
            angles: Dictionary of joint angles
            exercise_type: Type of exercise ('pushup', 'jumping_jack', or 'auto')
            frame_idx: Current frame index

        Returns:
            Dictionary with counts and state information
        """
        results = {
            'pushup': self.pushup_counter.process_frame(keypoints, angles, frame_idx),
            'jumping_jack': self.jumping_jack_counter.process_frame(keypoints, angles, frame_idx),
            'exercise_type': exercise_type
        }

        return results

    def get_counts(self) -> Dict[str, int]:
        """Get current counts for all exercise types."""
        return {
            'pushup': self.pushup_counter.count,
            'jumping_jack': self.jumping_jack_counter.count
        }

    def reset(self, exercise_type: Optional[str] = None):
        """
        Reset counter(s).

        Args:
            exercise_type: Type of exercise to reset ('pushup', 'jumping_jack', or None for all)
        """
        if exercise_type == 'pushup' or exercise_type is None:
            self.pushup_counter.reset()
        if exercise_type == 'jumping_jack' or exercise_type is None:
            self.jumping_jack_counter.reset()

    def set_exercise(self, exercise_type: str):
        """
        Set the active exercise type.

        Args:
            exercise_type: Type of exercise to count
        """
        logger.info(f"Setting active exercise to: {exercise_type}")
        # Reset all counters first
        self.reset()

        # This is for logging/visualization purposes
        self.active_exercise = exercise_type


# Factory function for creating counters
def create_counter(exercise_type: str, config: Optional[Dict] = None) -> any:
    """
    Factory function to create appropriate counter instance.

    Args:
        exercise_type: Type of counter ('pushup', 'jumping_jack', or 'all')
        config: Configuration dictionary

    Returns:
        Counter instance
    """
    config = config or {}

    if exercise_type == 'pushup':
        return PushUpCounter(**config)
    elif exercise_type == 'jumping_jack':
        return JumpingJackCounter(**config)
    elif exercise_type == 'all':
        return ExerciseCounter(config, config)
    else:
        raise ValueError(f"Unknown exercise type: {exercise_type}")
