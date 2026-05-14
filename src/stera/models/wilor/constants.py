"""MANO joint mappings and hand skeleton topology."""

from stera.annotations.hands import FINGER_NAMES, FINGER_JOINTS

# MANO 21-joint ordering:
#   0: wrist
#   1-4: thumb (mcp, pip, dip, tip)
#   5-8: index
#   9-12: middle
#   13-16: ring
#   17-20: pinky
FINGER_JOINT_SLICES = {
    "thumb": slice(1, 5),
    "index": slice(5, 9),
    "middle": slice(9, 13),
    "ring": slice(13, 17),
    "pinky": slice(17, 21),
}

# Skeleton connections for visualization (pairs of joint indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]

# Joint indices to try for depth anchoring (wrist first, then palm joints)
ANCHOR_JOINTS = [0, 9, 5, 13, 17]
