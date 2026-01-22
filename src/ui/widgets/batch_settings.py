"""Batch generation settings widget."""

from typing import Optional, Callable

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.core.gpu_manager import gpu_manager


class BatchSettingsWidget(Gtk.Box):
    """Widget for configuring batch generation settings."""

    LABEL_WIDTH = 55  # Match GenerationParamsWidget

    def __init__(self, on_changed: Optional[Callable[[], None]] = None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._on_changed = on_changed
        self._gpu_checkboxes: list[Gtk.CheckButton] = []

        self._build_ui()

    def _build_ui(self):
        """Build the widget UI."""
        # Header
        header = Gtk.Label(label="Batch")
        header.add_css_class("section-header")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Count row: Label | SpinButton
        count_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(count_row)

        count_label = Gtk.Label(label="Count")
        count_label.set_size_request(self.LABEL_WIDTH, -1)
        count_label.set_halign(Gtk.Align.START)
        count_label.add_css_class("caption")
        count_row.append(count_label)

        self._count_spin = Gtk.SpinButton.new_with_range(1, 100, 1)
        self._count_spin.set_value(1)
        self._count_spin.set_hexpand(True)
        self._count_spin.connect("value-changed", self._on_count_changed)
        count_row.append(self._count_spin)

        # GPU selection box (hidden by default, shown when count >= 2)
        self._gpu_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._gpu_box.set_visible(False)
        self.append(self._gpu_box)

        # GPU label
        gpu_label = Gtk.Label(label="GPUs")
        gpu_label.set_halign(Gtk.Align.START)
        gpu_label.add_css_class("caption")
        self._gpu_box.append(gpu_label)

        # GPU checkboxes container
        self._gpu_checkboxes_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._gpu_box.append(self._gpu_checkboxes_box)

        # Populate GPU checkboxes
        self._populate_gpu_checkboxes()

    def _populate_gpu_checkboxes(self):
        """Populate the GPU checkboxes based on available GPUs."""
        # Clear existing checkboxes
        while self._gpu_checkboxes_box.get_first_child():
            self._gpu_checkboxes_box.remove(self._gpu_checkboxes_box.get_first_child())
        self._gpu_checkboxes.clear()

        # Get available GPUs
        gpus = gpu_manager.get_all_gpus()

        for gpu in gpus:
            checkbox = Gtk.CheckButton(label=f"GPU {gpu.index}: {gpu.name}")
            checkbox.set_active(True)  # Enabled by default
            checkbox.add_css_class("caption")
            checkbox.connect("toggled", self._on_gpu_toggled)
            self._gpu_checkboxes_box.append(checkbox)
            self._gpu_checkboxes.append(checkbox)

    def _on_count_changed(self, spin: Gtk.SpinButton):
        """Handle count value change."""
        count = int(spin.get_value())
        # Show GPU selection when count >= 2
        self._gpu_box.set_visible(count >= 2)

        if self._on_changed:
            self._on_changed()

    def _on_gpu_toggled(self, checkbox: Gtk.CheckButton):
        """Handle GPU checkbox toggle."""
        if self._on_changed:
            self._on_changed()

    @property
    def batch_count(self) -> int:
        """Get the batch count."""
        return int(self._count_spin.get_value())

    @property
    def selected_gpu_indices(self) -> list[int]:
        """Get the indices of selected GPUs."""
        indices = []
        gpus = gpu_manager.get_all_gpus()
        for i, checkbox in enumerate(self._gpu_checkboxes):
            if checkbox.get_active() and i < len(gpus):
                indices.append(gpus[i].index)
        return indices

    @property
    def is_batch_mode(self) -> bool:
        """Check if batch mode is active (count > 1)."""
        return self.batch_count > 1

    def set_batch_count(self, count: int):
        """Set the batch count."""
        self._count_spin.set_value(max(1, min(100, count)))

    def set_gpu_selected(self, gpu_index: int, selected: bool):
        """Set whether a GPU is selected for batch generation."""
        gpus = gpu_manager.get_all_gpus()
        for i, gpu in enumerate(gpus):
            if gpu.index == gpu_index and i < len(self._gpu_checkboxes):
                self._gpu_checkboxes[i].set_active(selected)
                break

    def reset_to_defaults(self):
        """Reset to default settings."""
        self._count_spin.set_value(1)
        # Re-enable all GPU checkboxes
        for checkbox in self._gpu_checkboxes:
            checkbox.set_active(True)

    def refresh_gpus(self):
        """Refresh the GPU list."""
        self._populate_gpu_checkboxes()

    def set_sensitive_all(self, sensitive: bool):
        """Set sensitivity of all controls."""
        self._count_spin.set_sensitive(sensitive)
        for checkbox in self._gpu_checkboxes:
            checkbox.set_sensitive(sensitive)
