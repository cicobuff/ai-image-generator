"""LoRA selector widget for selecting and managing LoRAs."""

from pathlib import Path
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from src.ui.widgets.info_helper import SectionHeader, InfoButton, add_hover_tooltip, SECTION_INFO, LABEL_TOOLTIPS


@dataclass
class LoRASelection:
    """Represents a selected LoRA with its settings."""
    path: str  # Full path to the LoRA file
    name: str  # Display name
    weight: float  # LoRA weight (0.0 to 2.0)
    enabled: bool  # Whether this LoRA is active


class LoRAItemRow(Gtk.Box):
    """A single LoRA selection row with dropdown, weight, toggle, and remove button."""

    def __init__(
        self,
        lora_files: List[Tuple[str, str]],  # List of (path, name) tuples
        on_changed: Optional[Callable[[], None]] = None,
        on_remove: Optional[Callable[["LoRAItemRow"], None]] = None,
        show_remove: bool = True,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._lora_files = lora_files
        self._on_changed = on_changed
        self._on_remove = on_remove
        self._selected_path: str = ""
        self._selected_name: str = ""

        self.set_margin_top(2)
        self.set_margin_bottom(2)

        self._build_ui(show_remove)

    def _build_ui(self, show_remove: bool):
        """Build the row UI."""
        # Enable toggle
        self._enable_toggle = Gtk.CheckButton()
        self._enable_toggle.set_active(False)  # Default to disabled
        self._enable_toggle.set_tooltip_text("Enable/disable this LoRA")
        self._enable_toggle.connect("toggled", self._on_toggle_changed)
        self.append(self._enable_toggle)

        # LoRA dropdown
        self._dropdown = Gtk.DropDown()
        self._dropdown.set_hexpand(True)
        self._update_dropdown()
        self._dropdown.connect("notify::selected", self._on_selection_changed)
        self.append(self._dropdown)

        # Weight spin button
        self._weight_spin = Gtk.SpinButton.new_with_range(0.0, 2.0, 0.05)
        self._weight_spin.set_value(1.0)
        self._weight_spin.set_digits(2)
        self._weight_spin.set_width_chars(4)
        add_hover_tooltip(self._weight_spin, LABEL_TOOLTIPS["lora_weight"])
        self._weight_spin.connect("value-changed", self._on_weight_changed)
        self.append(self._weight_spin)

        # Remove button
        if show_remove:
            self._remove_button = Gtk.Button()
            self._remove_button.set_icon_name("list-remove-symbolic")
            self._remove_button.set_tooltip_text("Remove this LoRA")
            self._remove_button.add_css_class("flat")
            self._remove_button.connect("clicked", self._on_remove_clicked)
            self.append(self._remove_button)

    def _update_dropdown(self):
        """Update the dropdown with LoRA files."""
        # Create string list for dropdown
        string_list = Gtk.StringList()
        string_list.append("(none)")
        for _, name in self._lora_files:
            string_list.append(name)

        self._dropdown.set_model(string_list)
        self._dropdown.set_selected(0)

    def _on_selection_changed(self, dropdown, param):
        """Handle dropdown selection change."""
        selected = dropdown.get_selected()
        if selected == 0:
            self._selected_path = ""
            self._selected_name = ""
        else:
            idx = selected - 1
            if idx < len(self._lora_files):
                self._selected_path, self._selected_name = self._lora_files[idx]

        if self._on_changed:
            self._on_changed()

    def _on_toggle_changed(self, toggle):
        """Handle enable toggle change."""
        if self._on_changed:
            self._on_changed()

    def _on_weight_changed(self, spin):
        """Handle weight change."""
        if self._on_changed:
            self._on_changed()

    def _on_remove_clicked(self, button):
        """Handle remove button click."""
        if self._on_remove:
            self._on_remove(self)

    def update_lora_list(self, lora_files: List[Tuple[str, str]]):
        """Update the list of available LoRAs."""
        self._lora_files = lora_files
        # Store current selection
        current_path = self._selected_path

        # Update dropdown
        self._update_dropdown()

        # Restore selection if still valid
        if current_path:
            for i, (path, _) in enumerate(self._lora_files):
                if path == current_path:
                    self._dropdown.set_selected(i + 1)
                    return

        # Selection not found, reset
        self._selected_path = ""
        self._selected_name = ""

    def get_selection(self) -> Optional[LoRASelection]:
        """Get the current LoRA selection, or None if none selected."""
        if not self._selected_path:
            return None

        return LoRASelection(
            path=self._selected_path,
            name=self._selected_name,
            weight=self._weight_spin.get_value(),
            enabled=self._enable_toggle.get_active(),
        )

    @property
    def is_enabled(self) -> bool:
        """Check if this LoRA row is enabled."""
        return self._enable_toggle.get_active()

    @property
    def has_selection(self) -> bool:
        """Check if a LoRA is selected."""
        return bool(self._selected_path)


class LoRASelectorPanel(Gtk.Box):
    """Panel for selecting multiple LoRAs."""

    def __init__(
        self,
        on_changed: Optional[Callable[[], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._on_changed = on_changed
        self._lora_files: List[Tuple[str, str]] = []  # (path, name) tuples
        self._lora_rows: List[LoRAItemRow] = []

        self.add_css_class("lora-selector")
        self._build_ui()

    def _build_ui(self):
        """Build the panel UI."""
        # Header row with title and buttons
        header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(header_row)

        # Title
        title = Gtk.Label(label="LoRA")
        title.add_css_class("section-header")
        title.set_halign(Gtk.Align.START)
        header_row.append(title)

        # Info button
        info_button = InfoButton(SECTION_INFO["lora"])
        header_row.append(info_button)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        header_row.append(spacer)

        # Refresh button
        refresh_btn = Gtk.Button()
        refresh_btn.set_icon_name("view-refresh-symbolic")
        refresh_btn.set_tooltip_text("Refresh LoRA list")
        refresh_btn.add_css_class("flat")
        refresh_btn.connect("clicked", self._on_refresh_clicked)
        header_row.append(refresh_btn)

        # Add button
        add_btn = Gtk.Button()
        add_btn.set_icon_name("list-add-symbolic")
        add_btn.set_tooltip_text("Add another LoRA")
        add_btn.add_css_class("flat")
        add_btn.connect("clicked", self._on_add_clicked)
        header_row.append(add_btn)

        # Container for LoRA rows
        self._rows_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.append(self._rows_container)

        # Add initial row
        self._add_lora_row(show_remove=False)

    def _add_lora_row(self, show_remove: bool = True):
        """Add a new LoRA selector row."""
        row = LoRAItemRow(
            lora_files=self._lora_files,
            on_changed=self._on_lora_changed,
            on_remove=self._on_remove_row,
            show_remove=show_remove,
        )
        self._lora_rows.append(row)
        self._rows_container.append(row)

    def _on_add_clicked(self, button):
        """Handle add button click."""
        self._add_lora_row(show_remove=True)

    def _on_refresh_clicked(self, button):
        """Handle refresh button click."""
        self.scan_loras()

    def _on_remove_row(self, row: LoRAItemRow):
        """Handle remove row request."""
        if len(self._lora_rows) <= 1:
            # Don't remove the last row, just clear selection
            return

        self._lora_rows.remove(row)
        self._rows_container.remove(row)

        if self._on_changed:
            self._on_changed()

    def _on_lora_changed(self):
        """Handle LoRA selection change."""
        if self._on_changed:
            self._on_changed()

    def scan_loras(self, lora_dir: Optional[Path] = None):
        """Scan for LoRA files in the given directory."""
        if lora_dir is None:
            from src.core.config import config_manager
            models_dir = config_manager.config.get_models_path()
            lora_dir = models_dir / "loras"

        self._lora_files = []

        if lora_dir.exists():
            # Scan for LoRA files
            for ext in (".safetensors", ".pt", ".bin", ".ckpt"):
                for file in lora_dir.glob(f"*{ext}"):
                    if file.is_file():
                        name = file.stem
                        self._lora_files.append((str(file), name))

            # Sort by name
            self._lora_files.sort(key=lambda x: x[1].lower())

        # Update all rows
        for row in self._lora_rows:
            row.update_lora_list(self._lora_files)

    def get_active_loras(self) -> List[LoRASelection]:
        """Get list of active (enabled and selected) LoRAs."""
        loras = []
        for row in self._lora_rows:
            selection = row.get_selection()
            if selection and selection.enabled:
                loras.append(selection)
        return loras

    def get_all_selections(self) -> List[Optional[LoRASelection]]:
        """Get all selections (including None for unselected rows)."""
        return [row.get_selection() for row in self._lora_rows]

    @property
    def has_active_loras(self) -> bool:
        """Check if any LoRAs are active."""
        return len(self.get_active_loras()) > 0
