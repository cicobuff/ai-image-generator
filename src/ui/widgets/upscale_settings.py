"""Upscale settings widget with checkbox and model selector."""

from typing import Optional, Callable

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.core.model_manager import model_manager, ModelInfo, ModelType


class UpscaleSettingsWidget(Gtk.Box):
    """Widget for configuring upscale settings."""

    def __init__(self, on_changed: Optional[Callable[[], None]] = None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self._on_changed = on_changed
        self._models: list[ModelInfo] = []
        self._selected_model: Optional[ModelInfo] = None

        self._build_ui()

        # Register for model list updates
        model_manager.add_models_changed_callback(self._on_models_updated)

    def _build_ui(self):
        """Build the widget UI."""
        # Header
        header = Gtk.Label(label="Upscaling")
        header.add_css_class("section-header")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Enable checkbox
        checkbox_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._enable_checkbox = Gtk.CheckButton(label="Enable upscaling")
        self._enable_checkbox.connect("toggled", self._on_enable_toggled)
        checkbox_row.append(self._enable_checkbox)
        self.append(checkbox_row)

        # Model selector
        model_row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        model_row.set_margin_start(24)  # Indent under checkbox

        model_label = Gtk.Label(label="Upscale Model:")
        model_label.set_halign(Gtk.Align.START)
        model_row.append(model_label)

        # Dropdown row with refresh button
        dropdown_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        model_row.append(dropdown_row)

        self._model_dropdown = Gtk.DropDown()
        self._model_dropdown.set_hexpand(True)
        self._model_dropdown.set_sensitive(False)  # Disabled until checkbox is checked
        self._model_dropdown.connect("notify::selected", self._on_model_changed)
        dropdown_row.append(self._model_dropdown)

        # Refresh button
        self._refresh_button = Gtk.Button()
        self._refresh_button.set_icon_name("view-refresh-symbolic")
        self._refresh_button.set_tooltip_text("Refresh upscale model list")
        self._refresh_button.add_css_class("flat")
        self._refresh_button.connect("clicked", self._on_refresh_clicked)
        dropdown_row.append(self._refresh_button)

        self.append(model_row)
        self._model_row = model_row

        # Info label
        self._info_label = Gtk.Label(label="")
        self._info_label.set_halign(Gtk.Align.START)
        self._info_label.set_margin_start(24)
        self._info_label.add_css_class("dim-label")
        self.append(self._info_label)

        # Initialize model list
        self._update_model_list()

    def _update_model_list(self):
        """Update the dropdown with available upscale models."""
        self._models = model_manager.upscalers.copy()

        # Build string list for dropdown
        model_names = ["(None)"]
        for model in self._models:
            model_names.append(model.name)

        string_list = Gtk.StringList.new(model_names)
        self._model_dropdown.set_model(string_list)
        self._model_dropdown.set_selected(0)

        # Update info label
        if not self._models:
            self._info_label.set_text("No upscale models found in models/upscale/")
        else:
            self._info_label.set_text(f"{len(self._models)} upscale model(s) available")

    def _on_models_updated(self):
        """Called when the model list is updated."""
        self._update_model_list()

    def _on_enable_toggled(self, checkbox: Gtk.CheckButton):
        """Handle enable checkbox toggle."""
        enabled = checkbox.get_active()
        self._model_dropdown.set_sensitive(enabled)

        if self._on_changed:
            self._on_changed()

    def _on_model_changed(self, dropdown: Gtk.DropDown, param):
        """Handle model selection change."""
        selected = dropdown.get_selected()

        if selected == 0 or selected == Gtk.INVALID_LIST_POSITION:
            self._selected_model = None
        else:
            model_index = selected - 1
            if 0 <= model_index < len(self._models):
                self._selected_model = self._models[model_index]
            else:
                self._selected_model = None

        if self._on_changed:
            self._on_changed()

    @property
    def is_enabled(self) -> bool:
        """Check if upscaling is enabled."""
        return self._enable_checkbox.get_active()

    @property
    def selected_model(self) -> Optional[ModelInfo]:
        """Get the selected upscale model."""
        if not self.is_enabled:
            return None
        return self._selected_model

    @property
    def selected_model_name(self) -> str:
        """Get the name of the selected upscale model."""
        if self._selected_model:
            return self._selected_model.name
        return ""

    @property
    def selected_model_path(self) -> Optional[str]:
        """Get the path of the selected upscale model."""
        if self._selected_model:
            return str(self._selected_model.path)
        return None

    def set_enabled(self, enabled: bool):
        """Set whether upscaling is enabled."""
        self._enable_checkbox.set_active(enabled)

    def set_model_by_name(self, name: str):
        """Select an upscale model by its name."""
        if not name:
            self._model_dropdown.set_selected(0)
            return

        for i, model in enumerate(self._models):
            if model.name == name:
                self._model_dropdown.set_selected(i + 1)  # +1 for "None" option
                return

        # Model not found, select None
        self._model_dropdown.set_selected(0)

    def reset_to_defaults(self):
        """Reset to default settings (disabled, no model selected)."""
        self._enable_checkbox.set_active(False)
        self._model_dropdown.set_selected(0)

    def _on_refresh_clicked(self, button: Gtk.Button):
        """Handle refresh button click - rescan models."""
        import threading
        from gi.repository import GLib

        # Disable button during scan
        button.set_sensitive(False)

        def scan_thread():
            model_manager.scan_models()
            GLib.idle_add(restore_button)

        def restore_button():
            button.set_sensitive(True)

        thread = threading.Thread(target=scan_thread, daemon=True)
        thread.start()
