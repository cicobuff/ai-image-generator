"""GPU monitoring display widget with VRAM, utilization, and temperature."""

import threading
from typing import Optional

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from src.core.gpu_manager import gpu_manager, GPUInfo
from src.core.config import config_manager
from src.utils.constants import GPU_MEMORY_UPDATE_INTERVAL
from src.ui.widgets.info_helper import SectionHeader, SECTION_INFO


class GPUUsageIndicator(Gtk.DrawingArea):
    """Small 10x10 pixel box showing GPU utilization."""

    def __init__(self):
        super().__init__()
        self._utilization = 0
        self.set_size_request(10, 10)
        self.set_draw_func(self._on_draw)

    def set_utilization(self, percent: int):
        """Set the utilization percentage (0-100)."""
        self._utilization = max(0, min(100, percent))
        self.queue_draw()

    def _on_draw(self, area, cr, width, height):
        """Draw the utilization indicator."""
        # Draw dark background
        cr.set_source_rgb(0.16, 0.16, 0.16)  # #292929
        cr.rectangle(0, 0, width, height)
        cr.fill()

        # Draw filled portion from bottom
        if self._utilization > 0:
            # Calculate fill height based on utilization
            fill_height = (self._utilization / 100.0) * height
            cr.set_source_rgb(0.0, 0.5, 0.0)  # Dark green
            cr.rectangle(0, height - fill_height, width, fill_height)
            cr.fill()

        # Draw border
        cr.set_source_rgb(0.3, 0.3, 0.3)
        cr.set_line_width(1)
        cr.rectangle(0.5, 0.5, width - 1, height - 1)
        cr.stroke()


class VRAMBar(Gtk.Box):
    """Single GPU monitoring bar with VRAM, utilization, and temperature."""

    def __init__(self, gpu_index: int):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._gpu_index = gpu_index
        self._gpu_info: Optional[GPUInfo] = None

        self._build_ui()

    def _build_ui(self):
        """Build the monitoring bar UI."""
        # GPU label
        self._label = Gtk.Label()
        self._label.set_halign(Gtk.Align.START)
        self._label.add_css_class("vram-label")
        self.append(self._label)

        # Horizontal box for progress bar and indicators
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.append(hbox)

        # Progress bar for VRAM
        self._progress = Gtk.ProgressBar()
        self._progress.add_css_class("vram-bar")
        self._progress.set_show_text(True)
        self._progress.set_hexpand(True)
        hbox.append(self._progress)

        # GPU usage indicator box
        usage_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        hbox.append(usage_box)

        self._usage_indicator = GPUUsageIndicator()
        usage_box.append(self._usage_indicator)

        self._usage_label = Gtk.Label()
        self._usage_label.add_css_class("caption")
        self._usage_label.add_css_class("monospace")
        self._usage_label.set_width_chars(4)
        self._usage_label.set_xalign(1.0)  # Right align
        usage_box.append(self._usage_label)

        # Temperature display
        self._temp_label = Gtk.Label()
        self._temp_label.add_css_class("caption")
        self._temp_label.add_css_class("monospace")
        self._temp_label.set_width_chars(5)
        self._temp_label.set_xalign(1.0)  # Right align
        hbox.append(self._temp_label)

        # Initial update
        self.update_from_info(None)

    def update_from_info(self, info: Optional[GPUInfo]):
        """Update the display from GPUInfo."""
        self._gpu_info = info

        if info is None:
            self._label.set_text(f"GPU {self._gpu_index}: Not available")
            self._progress.set_fraction(0)
            self._progress.set_text("N/A")
            self._usage_indicator.set_utilization(0)
            self._usage_label.set_text("  --")
            self._temp_label.set_text("  --")
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

        # Update GPU utilization (pad to 4 chars: "  0%" to "100%")
        self._usage_indicator.set_utilization(info.utilization)
        self._usage_label.set_text(f"{info.utilization:3d}%")

        # Update temperature (pad to 5 chars: " 30°C" to "100°C")
        self._temp_label.set_text(f"{info.temperature:3d}°C")


class MonitoringDisplay(Gtk.Box):
    """Widget displaying GPU monitoring for selected GPUs."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self._bars: dict[int, VRAMBar] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        self._build_ui()
        self._start_monitoring_thread()

    def _build_ui(self):
        """Build the monitoring display UI."""
        # Header with info button
        header = SectionHeader("Monitoring", SECTION_INFO["monitoring"])
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
            self._bars[gpu_index] = bar
            self._bars_container.append(bar)

        # If no GPUs selected, show message
        if not selected_gpus:
            label = Gtk.Label(label="No GPUs selected")
            label.add_css_class("dim-label")
            self._bars_container.append(label)

    def _start_monitoring_thread(self):
        """Start the monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

    def _stop_monitoring_thread(self):
        """Stop the monitoring thread."""
        self._stop_monitoring.set()
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=1.0)
            self._monitoring_thread = None

    def _monitoring_loop(self):
        """Background thread loop for monitoring GPUs."""
        while not self._stop_monitoring.is_set():
            # Collect GPU info in background thread
            gpu_infos = {}
            selected_gpus = config_manager.config.gpus.selected
            for gpu_index in selected_gpus:
                info = gpu_manager.get_gpu_info(gpu_index)
                if info:
                    gpu_infos[gpu_index] = info

            # Schedule UI update on main thread
            GLib.idle_add(self._update_ui_from_thread, gpu_infos)

            # Wait for next update interval
            self._stop_monitoring.wait(GPU_MEMORY_UPDATE_INTERVAL / 1000.0)

    def _update_ui_from_thread(self, gpu_infos: dict[int, GPUInfo]) -> bool:
        """Update UI from monitoring thread data. Called on main thread."""
        for gpu_index, bar in self._bars.items():
            info = gpu_infos.get(gpu_index)
            bar.update_from_info(info)
        return False  # Don't repeat

    def refresh_config(self):
        """Refresh when config changes."""
        self._update_bars()

    def do_unrealize(self):
        """Clean up when widget is unrealized."""
        self._stop_monitoring_thread()
        Gtk.Box.do_unrealize(self)


# Backward compatibility alias
VRAMDisplay = MonitoringDisplay
