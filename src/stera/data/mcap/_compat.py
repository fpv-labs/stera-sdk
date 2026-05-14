"""Monkey-patch mcap_ros2 to handle three-part type names (e.g. std_msgs/msg/Header)."""

import re


def patch_mcap_ros2():
    """Fix mcap_ros2 parser for schemas using 'pkg/msg/Type' format."""
    import mcap_ros2._dynamic as _dyn

    if getattr(_dyn._for_each_msgdef, "_fpv_patched", False):
        return

    _orig = _dyn._for_each_msgdef

    def _patched(schema_name, schema_text, fn):
        schema_text = re.sub(r"(\b\w+)/msg/(\w+)", r"\1/\2", schema_text)
        return _orig(schema_name, schema_text, fn)

    _patched._fpv_patched = True
    _dyn._for_each_msgdef = _patched
