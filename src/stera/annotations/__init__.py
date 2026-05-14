"""Annotation types: camera pose, hand tracking, body pose."""

from stera.annotations.pose.camera import CameraPose
from stera.annotations.pose.body import UpperBodyPose, UPPER_BODY_JOINTS, UPPER_BODY_CONNECTIONS
from stera.annotations.hands import HandPose, FINGER_NAMES, FINGER_JOINTS

__all__ = [
    "CameraPose",
    "UpperBodyPose",
    "UPPER_BODY_JOINTS",
    "UPPER_BODY_CONNECTIONS",
    "HandPose",
    "FINGER_NAMES",
    "FINGER_JOINTS",
]
