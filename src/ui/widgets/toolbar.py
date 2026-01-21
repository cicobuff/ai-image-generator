"""Main toolbar widget with action buttons."""

from typing import Optional, Callable

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.core.generation_service import GenerationState


class Toolbar(Gtk.Box):
    """Main toolbar with Load, Clear, and Generate buttons."""

    def __init__(
        self,
        on_load: Optional[Callable[[], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        on_generate: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._on_load = on_load
        self._on_clear = on_clear
        self._on_generate = on_generate
        self._on_cancel = on_cancel

        self.add_css_class("toolbar")
        self._build_ui()

    def _build_ui(self):
        """Build the toolbar UI."""
        # Load Models button
        self._load_button = Gtk.Button(label="Load Models")
        self._load_button.connect("clicked", self._on_load_clicked)
        self.append(self._load_button)

        # Clear Models button
        self._clear_button = Gtk.Button(label="Clear Models")
        self._clear_button.connect("clicked", self._on_clear_clicked)
        self.append(self._clear_button)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        separator.set_margin_start(8)
        separator.set_margin_end(8)
        self.append(separator)

        # Generate button
        self._generate_button = Gtk.Button(label="Generate Image")
        self._generate_button.add_css_class("suggested-action")
        self._generate_button.connect("clicked", self._on_generate_clicked)
        self.append(self._generate_button)

        # Cancel button (hidden by default)
        self._cancel_button = Gtk.Button(label="Cancel")
        self._cancel_button.add_css_class("destructive-action")
        self._cancel_button.connect("clicked", self._on_cancel_clicked)
        self._cancel_button.set_visible(False)
        self.append(self._cancel_button)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        self.append(spacer)

        # Progress label
        self._progress_label = Gtk.Label(label="")
        self._progress_label.add_css_class("dim-label")
        self.append(self._progress_label)

        # Progress bar
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_size_request(200, -1)
        self._progress_bar.set_visible(False)
        self.append(self._progress_bar)

    def _on_load_clicked(self, button):
        """Handle Load button click."""
        if self._on_load:
            self._on_load()

    def _on_clear_clicked(self, button):
        """Handle Clear button click."""
        if self._on_clear:
            self._on_clear()

    def _on_generate_clicked(self, button):
        """Handle Generate button click."""
        if self._on_generate:
            self._on_generate()

    def _on_cancel_clicked(self, button):
        """Handle Cancel button click."""
        if self._on_cancel:
            self._on_cancel()

    def set_state(self, state: GenerationState):
        """Update toolbar state based on generation state."""
        if state == GenerationState.IDLE:
            self._load_button.set_sensitive(True)
            self._clear_button.set_sensitive(True)
            self._generate_button.set_sensitive(True)
            self._generate_button.set_visible(True)
            self._cancel_button.set_visible(False)
            self._progress_bar.set_visible(False)

        elif state == GenerationState.LOADING:
            self._load_button.set_sensitive(False)
            self._clear_button.set_sensitive(False)
            self._generate_button.set_sensitive(False)
            self._generate_button.set_visible(True)
            self._cancel_button.set_visible(False)
            self._progress_bar.set_visible(True)

        elif state == GenerationState.GENERATING:
            self._load_button.set_sensitive(False)
            self._clear_button.set_sensitive(False)
            self._generate_button.set_visible(False)
            self._cancel_button.set_visible(True)
            self._progress_bar.set_visible(True)

        elif state == GenerationState.CANCELLING:
            self._load_button.set_sensitive(False)
            self._clear_button.set_sensitive(False)
            self._generate_button.set_visible(False)
            self._cancel_button.set_sensitive(False)
            self._progress_bar.set_visible(True)

    def set_model_loaded(self, loaded: bool):
        """Update state based on whether a model is loaded."""
        self._generate_button.set_sensitive(loaded)
        self._clear_button.set_sensitive(loaded)

    def set_progress(self, message: str, fraction: float):
        """Update progress display."""
        self._progress_label.set_text(message)
        self._progress_bar.set_fraction(fraction)

    def set_step_progress(self, step: int, total: int):
        """Update step progress."""
        if total > 0:
            fraction = step / total
            self._progress_bar.set_fraction(fraction)
            self._progress_label.set_text(f"Step {step}/{total}")

    def clear_progress(self):
        """Clear progress display."""
        self._progress_label.set_text("")
        self._progress_bar.set_fraction(0)
        self._progress_bar.set_visible(False)
