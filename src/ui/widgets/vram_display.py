"""VRAM usage display widget."""

from typing import Optional

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from src.core.gpu_manager import gpu_manager, GPUInfo
from src.core.config import config_manager
from src.utils.constants import GPU_MEMORY_UPDATE_INTERVAL


class VRAMBar(Gtk.Box):
    """Single GPU VRAM usage bar."""

    def __init__(self, gpu_index: int):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._gpu_index = gpu_index

        self._build_ui()

    def _build_ui(self):
        """Build the VRAM bar UI."""
        # GPU label
        self._label = Gtk.Label()
        self._label.set_halign(Gtk.Align.START)
        self._label.add_css_class("vram-label")
        self.append(self._label)

        # Progress bar
        self._progress = Gtk.ProgressBar()
        self._progress.add_css_class("vram-bar")
        self._progress.set_show_text(True)
        self.append(self._progress)

        # Initial update
        self.update()

    def update(self):
        """Update the VRAM display."""
        info = gpu_manager.get_gpu_info(self._gpu_index)

        if info is None:
            self._label.set_text(f"GPU {self._gpu_index}: Not available")
            self._progress.set_fraction(0)
            self._progress.set_text("N/A")
            return

        # Update label
        self._label.set_text(f"GPU {self._gpu_index}: {info.name}")

        # Update progress bar
        fraction = info.used_memory / info.total_memory if info.total_memory > 0 else 0
        self._progress.set_fraction(fraction)
        self._progress.set_text(
            f"{info.used_memory_gb:.1f} / {info.total_memory_gb:.1f} GB"
        )

        # Update color based on usage
        self._progress.remove_css_class("vram-low")
        self._progress.remove_css_class("vram-medium")
        self._progress.remove_css_class("vram-high")

        if fraction < 0.6:
            self._progress.add_css_class("vram-low")
        elif fraction < 0.85:
            self._progress.add_css_class("vram-medium")
        else:
            self._progress.add_css_class("vram-high")


class VRAMDisplay(Gtk.Box):
    """Widget displaying VRAM usage for selected GPUs."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self._bars: list[VRAMBar] = []
        self._update_timer: Optional[int] = None

        self._build_ui()
        self._start_update_timer()

    def _build_ui(self):
        """Build the VRAM display UI."""
        # Header
        header = Gtk.Label(label="VRAM Usage")
        header.add_css_class("section-header")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Container for bars
        self._bars_container = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=8
        )
        self.append(self._bars_container)

        # Create bars for selected GPUs
        self._update_bars()

    def _update_bars(self):
        """Update the GPU bars based on config."""
        # Clear existing bars
        while self._bars_container.get_first_child():
            self._bars_container.remove(self._bars_container.get_first_child())
        self._bars.clear()

        # Create bars for selected GPUs
        selected_gpus = config_manager.config.gpus.selected
        for gpu_index in selected_gpus:
            bar = VRAMBar(gpu_index)
            self._bars.append(bar)
            self._bars_container.append(bar)

        # If no GPUs selected, show message
        if not selected_gpus:
            label = Gtk.Label(label="No GPUs selected")
            label.add_css_class("dim-label")
            self._bars_container.append(label)

    def _start_update_timer(self):
        """Start the periodic update timer."""
        if self._update_timer is None:
            self._update_timer = GLib.timeout_add(
                GPU_MEMORY_UPDATE_INTERVAL, self._on_update_timeout
            )

    def _stop_update_timer(self):
        """Stop the periodic update timer."""
        if self._update_timer is not None:
            GLib.source_remove(self._update_timer)
            self._update_timer = None

    def _on_update_timeout(self) -> bool:
        """Handle update timer tick."""
        self.update()
        return True  # Continue timer

    def update(self):
        """Update all VRAM bars."""
        for bar in self._bars:
            bar.update()

    def refresh_config(self):
        """Refresh when config changes."""
        self._update_bars()

    def do_unrealize(self):
        """Clean up when widget is unrealized."""
        self._stop_update_timer()
        Gtk.Box.do_unrealize(self)
