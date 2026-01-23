"""Model selector dropdown widget."""

from typing import Optional, Callable

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GObject

from src.core.model_manager import model_manager, ModelInfo, ModelType
from src.ui.widgets.info_helper import add_hover_tooltip, LABEL_TOOLTIPS


class ModelSelector(Gtk.Box):
    """Dropdown widget for selecting models with embedded component indicators."""

    def __init__(
        self,
        label: str,
        model_type: ModelType,
        on_changed: Optional[Callable[[Optional[ModelInfo]], None]] = None,
        compact: bool = False,
        label_width: int = 70,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._model_type = model_type
        self._on_changed = on_changed
        self._models: list[ModelInfo] = []
        self._compact = compact
        self._label_width = label_width

        self.add_css_class("model-selector")
        self._build_ui(label)

        # Register for model list updates
        model_manager.add_models_changed_callback(self._on_models_updated)

    def _build_ui(self, label_text: str):
        """Build the widget UI."""
        # Label with fixed width for alignment
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_size_request(self._label_width, -1)
        if self._compact:
            label.add_css_class("caption")

        # Add tooltip based on model type
        tooltip_key = label_text.lower()
        if tooltip_key in LABEL_TOOLTIPS:
            add_hover_tooltip(label, LABEL_TOOLTIPS[tooltip_key])

        self.append(label)

        # Dropdown
        self._dropdown = Gtk.DropDown()
        self._dropdown.set_hexpand(True)
        self._dropdown.connect("notify::selected", self._on_selection_changed)
        self.append(self._dropdown)

        # Refresh button
        self._refresh_button = Gtk.Button()
        self._refresh_button.set_icon_name("view-refresh-symbolic")
        self._refresh_button.set_tooltip_text("Refresh model list")
        self._refresh_button.add_css_class("flat")
        self._refresh_button.connect("clicked", self._on_refresh_clicked)
        self.append(self._refresh_button)

        # Initialize with empty model
        self._update_model_list()

    def _update_model_list(self):
        """Update the dropdown with available models."""
        # Get models based on type
        if self._model_type == ModelType.CHECKPOINT:
            self._models = model_manager.checkpoints.copy()
        elif self._model_type == ModelType.VAE:
            self._models = model_manager.vaes.copy()
        elif self._model_type == ModelType.CLIP:
            self._models = model_manager.clips.copy()

        # Build string list for dropdown
        # Add "None" option at the start
        model_names = ["(None)"]
        for model in self._models:
            model_names.append(model.display_name)

        string_list = Gtk.StringList.new(model_names)
        self._dropdown.set_model(string_list)
        self._dropdown.set_selected(0)

    def _on_models_updated(self):
        """Called when the model list is updated."""
        self._update_model_list()

    def _on_selection_changed(self, dropdown: Gtk.DropDown, param):
        """Handle dropdown selection change."""
        selected = dropdown.get_selected()

        if selected == 0 or selected == Gtk.INVALID_LIST_POSITION:
            # None selected
            model = None
        else:
            # Adjust for "None" option at index 0
            model_index = selected - 1
            if 0 <= model_index < len(self._models):
                model = self._models[model_index]
            else:
                model = None

        # Update model manager selection
        if self._model_type == ModelType.CHECKPOINT:
            model_manager.select_checkpoint(model)
        elif self._model_type == ModelType.VAE:
            model_manager.select_vae(model)
        elif self._model_type == ModelType.CLIP:
            model_manager.select_clip(model)

        # Notify callback
        if self._on_changed:
            self._on_changed(model)

    def get_selected(self) -> Optional[ModelInfo]:
        """Get the currently selected model."""
        selected = self._dropdown.get_selected()
        if selected == 0 or selected == Gtk.INVALID_LIST_POSITION:
            return None
        model_index = selected - 1
        if 0 <= model_index < len(self._models):
            return self._models[model_index]
        return None

    def set_selected_by_name(self, name: str):
        """Select a model by its name."""
        for i, model in enumerate(self._models):
            if model.name == name:
                self._dropdown.set_selected(i + 1)  # +1 for "None" option
                return
        self._dropdown.set_selected(0)

    def refresh(self):
        """Refresh the model list."""
        self._update_model_list()

    def _on_refresh_clicked(self, button: Gtk.Button):
        """Handle refresh button click - rescan models."""
        import threading
        from gi.repository import GLib

        # Disable button during scan
        button.set_sensitive(False)

        def scan_thread():
            model_manager.scan_models()
            GLib.idle_add(self._on_scan_complete)

        def restore_button():
            button.set_sensitive(True)

        self._on_scan_complete = restore_button

        thread = threading.Thread(target=scan_thread, daemon=True)
        thread.start()
