"""Minimal napari dock widget scaffold for previewing detections.

Provides a `preview_detections` magicgui that finds the first image layer
in the active viewer, runs `detect.detect_frame` on the current slice,
and adds/updates a `det_preview` points layer.

This file is a lightweight scaffold — no plugin registration required to
use it from a notebook: import and call `preview_detections.show()`.
"""
from typing import Optional

import numpy as np

try:
    from magicgui import magicgui
except Exception:  # pragma: no cover - optional import
    def magicgui(*a, **k):
        raise ImportError("magicgui is required for the napari widget scaffold")

import napari

from detect import detect_frame


@magicgui(call_button="Preview",
         diameter={'min': 1, 'max': 50, 'step': 1},
         separation={'min': 1, 'max': 50, 'step': 1},
         minmass={'label': 'minmass', 'widget_type': 'FloatSpin', 'min': 0.0, 'max': 1e7, 'step': 1.0, 'nullable': True})
def preview_detections(viewer: napari.Viewer, diameter: int = 5, separation: int = 3, minmass: Optional[float] = None):
    """Run detection on the first image layer / current frame and show preview.

    Usage from a notebook or Python:
    - ensure `%gui qt` is active and a napari Viewer exists
    - `widget = preview_detections()`
    - `widget.show()` to display the dock
    """
    # find first image layer
    image_layer = None
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image):
            image_layer = layer
            break
    if image_layer is None:
        raise RuntimeError("No image layer found in the viewer")

    data = image_layer.data
    # support (t, y, x) or (y, x)
    if data.ndim == 3:
        # get current step for first axis if available
        try:
            t = viewer.dims.current_step[0]
        except Exception:
            t = 0
        frame = np.asarray(data[t])
    else:
        frame = np.asarray(data)

    pts = detect_frame(frame, method='trackpy', diameter=diameter, separation=separation, minmass=minmass)

    # remove existing preview layer if present
    if 'det_preview' in viewer.layers:
        try:
            viewer.layers.remove('det_preview')
        except Exception:
            pass

    viewer.add_points(pts[['x','y']].values, name='det_preview', face_color='green', size=5)


def create_and_show(viewer: napari.Viewer):
    """Helper to create the widget and show it as a docked widget.
    Call `create_and_show(viewer)` from a running notebook with an active Viewer.
    """
    w = preview_detections(gui=False)
    w.native.setWindowTitle('Detection Preview')
    viewer.window.add_dock_widget(w.native, area='right')
