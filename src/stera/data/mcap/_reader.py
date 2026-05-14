"""Low-level MCAP reader with ROS2 CDR decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from stera.data.mcap._compat import patch_mcap_ros2

# Apply monkey-patch before any mcap_ros2 decoding
patch_mcap_ros2()


@dataclass
class TopicConfig:
    """Maps semantic names to MCAP topic strings."""

    rgb: str = "/camera/rgb/compressed"
    depth: str = "/camera/depth"
    camera_pose: str = "/camera/pose"
    rgb_camera_info: str = "/camera/camera_info"
    depth_camera_info: str = "/camera/depth/camera_info"
    imu: str = "/device/imu"
    tracking_state: str = "/camera/tracking_state"
    tf: str = "/tf"
    point_cloud: str = "/map/point_cloud"
    mesh: str = "/map/mesh"
    mesh_cloud: str = "/map/mesh_cloud"
    trajectory: str = "/trajectory"


class MCAPRawReader:
    """Low-level iterator over decoded MCAP messages."""

    def __init__(self, path: str | Path, topics: TopicConfig | None = None):
        self.path = Path(path)
        self.topics = topics or TopicConfig()
        if not self.path.exists():
            raise FileNotFoundError(f"MCAP file not found: {self.path}")

    def _make_reader(self, fh):
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory as RosDecoder
        return make_reader(fh, decoder_factories=[RosDecoder()])

    def summary(self) -> dict:
        """Return topic names, message counts, duration."""
        with open(self.path, "rb") as f:
            reader = self._make_reader(f)
            s = reader.get_summary()
            stats = s.statistics
            channels = {cid: ch.topic for cid, ch in s.channels.items()}
            counts = {channels.get(cid, str(cid)): cnt for cid, cnt in stats.channel_message_counts.items()}
            duration_ns = stats.message_end_time - stats.message_start_time
            return {
                "topics": counts,
                "message_count": stats.message_count,
                "duration_sec": duration_ns / 1e9,
                "start_time": stats.message_start_time / 1e9,
                "end_time": stats.message_end_time / 1e9,
            }

    def iter_topic(self, topic: str) -> Iterator[tuple[float, Any]]:
        """Yield (timestamp_sec, decoded_msg) for a single topic."""
        with open(self.path, "rb") as f:
            reader = self._make_reader(f)
            for _schema, _channel, message, decoded in reader.iter_decoded_messages(topics=[topic]):
                ts = message.log_time / 1e9
                yield ts, decoded

    def read_first(self, topic: str) -> Any | None:
        """Read and return the first decoded message on a topic."""
        for _ts, msg in self.iter_topic(topic):
            return msg
        return None

    def read_last(self, topic: str) -> Any | None:
        """Read and return the last decoded message on a topic."""
        last = None
        for _ts, msg in self.iter_topic(topic):
            last = msg
        return last
