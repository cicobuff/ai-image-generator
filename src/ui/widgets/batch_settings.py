"""Batch generation settings widget."""

from typing import Optional, Callable

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.ui.widgets.info_helper import SectionHeader, add_hover_tooltip, SECTION_INFO, LABEL_TOOLTIPS


class BatchSettingsWidget(Gtk.Box):
    """Widget for configuring batch generation settings."""

    LABEL_WIDTH = 55  # Match GenerationParamsWidget

    def __init__(self, on_changed: Optional[Callable[[], None]] = None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._on_changed = on_changed

        self._build_ui()

    def _build_ui(self):
        """Build the widget UI."""
        # Header with info button
        header = SectionHeader("Batch", SECTION_INFO["batch"])
        self.append(header)

        # Count row: Label | SpinButton
        count_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(count_row)

        count_label = Gtk.Label(label="Count")
        count_label.set_size_request(self.LABEL_WIDTH, -1)
        count_label.set_halign(Gtk.Align.START)
        count_label.add_css_class("caption")
        add_hover_tooltip(count_label, LABEL_TOOLTIPS["batch_count"])
        count_row.append(count_label)

        self._count_spin = Gtk.SpinButton.new_with_range(1, 1000, 1)
        self._count_spin.set_value(1)
        self._count_spin.set_hexpand(True)
        self._count_spin.connect("value-changed", self._on_count_changed)
        count_row.append(self._count_spin)

    def _on_count_changed(self, spin: Gtk.SpinButton):
        """Handle count value change."""
        if self._on_changed:
            self._on_changed()

    @property
    def batch_count(self) -> int:
        """Get the batch count."""
        return int(self._count_spin.get_value())

    @property
    def is_batch_mode(self) -> bool:
        """Check if batch mode is active (count > 1)."""
        return self.batch_count > 1

    def set_batch_count(self, count: int):
        """Set the batch count."""
        self._count_spin.set_value(max(1, min(1000, count)))

    def reset_to_defaults(self):
        """Reset to default settings."""
        self._count_spin.set_value(1)

    def set_sensitive_all(self, sensitive: bool):
        """Set sensitivity of all controls."""
        self._count_spin.set_sensitive(sensitive)
