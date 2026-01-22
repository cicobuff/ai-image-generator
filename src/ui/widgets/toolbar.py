"""Main toolbar widget with action buttons."""

from typing import Optional, Callable
from enum import Enum

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.core.generation_service import GenerationState


class InpaintTool(Enum):
    """Inpaint mask drawing tools."""
    NONE = "none"
    RECT = "rect"
    PAINT = "paint"


class Toolbar(Gtk.Box):
    """Main toolbar with Load, Clear, and Generate buttons."""

    def __init__(
        self,
        on_load: Optional[Callable[[], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        on_generate: Optional[Callable[[], None]] = None,
        on_img2img: Optional[Callable[[], None]] = None,
        on_upscale: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        on_inpaint_mode_changed: Optional[Callable[[bool], None]] = None,
        on_inpaint_tool_changed: Optional[Callable[[InpaintTool], None]] = None,
        on_clear_masks: Optional[Callable[[], None]] = None,
        on_generate_inpaint: Optional[Callable[[], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._on_load = on_load
        self._on_clear = on_clear
        self._on_generate = on_generate
        self._on_img2img = on_img2img
        self._on_upscale = on_upscale
        self._on_cancel = on_cancel
        self._on_inpaint_mode_changed = on_inpaint_mode_changed
        self._on_inpaint_tool_changed = on_inpaint_tool_changed
        self._on_clear_masks = on_clear_masks
        self._on_generate_inpaint = on_generate_inpaint

        self._inpaint_mode = False
        self._current_tool = InpaintTool.NONE
        self._model_loaded = False
        self._upscale_enabled = False

        self.add_css_class("toolbar")
        self._build_ui()
        self._update_generation_buttons_sensitivity()

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
        self._generate_button = Gtk.Button(label="Generate")
        self._generate_button.add_css_class("suggested-action")
        self._generate_button.connect("clicked", self._on_generate_clicked)
        self.append(self._generate_button)

        # Image to Image button
        self._img2img_button = Gtk.Button(label="Image to Image")
        self._img2img_button.set_tooltip_text("Generate a new image based on the current image")
        self._img2img_button.connect("clicked", self._on_img2img_clicked)
        self.append(self._img2img_button)

        # Upscale button
        self._upscale_button = Gtk.Button(label="Upscale")
        self._upscale_button.set_tooltip_text("Upscale the current image using the selected upscaler")
        self._upscale_button.connect("clicked", self._on_upscale_clicked)
        self._upscale_button.set_sensitive(False)  # Disabled by default
        self.append(self._upscale_button)

        # Separator before inpaint
        self._inpaint_separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self._inpaint_separator.set_margin_start(8)
        self._inpaint_separator.set_margin_end(8)
        self._inpaint_separator.set_visible(False)
        self.append(self._inpaint_separator)

        # Inpaint Mode toggle
        self._inpaint_toggle = Gtk.ToggleButton(label="Inpaint Mode")
        self._inpaint_toggle.set_tooltip_text("Enable inpainting mode to draw masks")
        self._inpaint_toggle.connect("toggled", self._on_inpaint_toggled)
        self._inpaint_toggle.set_visible(False)
        self.append(self._inpaint_toggle)

        # Rect Mask tool button (visible in inpaint mode)
        self._rect_mask_button = Gtk.ToggleButton(label="Rect Mask")
        self._rect_mask_button.set_tooltip_text("Draw rectangular masks")
        self._rect_mask_button.connect("toggled", self._on_rect_mask_toggled)
        self._rect_mask_button.set_visible(False)
        self.append(self._rect_mask_button)

        # Paint Mask tool button (visible in inpaint mode)
        self._paint_mask_button = Gtk.ToggleButton(label="Paint Mask")
        self._paint_mask_button.set_tooltip_text("Paint masks with brush (25px radius)")
        self._paint_mask_button.connect("toggled", self._on_paint_mask_toggled)
        self._paint_mask_button.set_visible(False)
        self.append(self._paint_mask_button)

        # Clear Masks button (visible in inpaint mode)
        self._clear_masks_button = Gtk.Button(label="Clear Masks")
        self._clear_masks_button.set_tooltip_text("Clear all drawn masks")
        self._clear_masks_button.connect("clicked", self._on_clear_masks_clicked)
        self._clear_masks_button.set_visible(False)
        self.append(self._clear_masks_button)

        # Generate Inpaint button (visible in inpaint mode)
        self._generate_inpaint_button = Gtk.Button(label="Generate Inpaint")
        self._generate_inpaint_button.add_css_class("suggested-action")
        self._generate_inpaint_button.set_tooltip_text("Generate inpainted image in masked areas")
        self._generate_inpaint_button.connect("clicked", self._on_generate_inpaint_clicked)
        self._generate_inpaint_button.set_visible(False)
        self.append(self._generate_inpaint_button)

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

    def _on_img2img_clicked(self, button):
        """Handle Image to Image button click."""
        if self._on_img2img:
            self._on_img2img()

    def _on_upscale_clicked(self, button):
        """Handle Upscale button click."""
        if self._on_upscale:
            self._on_upscale()

    def _on_cancel_clicked(self, button):
        """Handle Cancel button click."""
        if self._on_cancel:
            self._on_cancel()

    def _on_inpaint_toggled(self, button):
        """Handle Inpaint Mode toggle."""
        self._inpaint_mode = button.get_active()
        self._update_inpaint_ui()
        if self._on_inpaint_mode_changed:
            self._on_inpaint_mode_changed(self._inpaint_mode)
        # Reset tool when exiting inpaint mode
        if not self._inpaint_mode:
            self._current_tool = InpaintTool.NONE
            self._rect_mask_button.set_active(False)
            self._paint_mask_button.set_active(False)
            if self._on_inpaint_tool_changed:
                self._on_inpaint_tool_changed(InpaintTool.NONE)

    def _on_rect_mask_toggled(self, button):
        """Handle Rect Mask tool toggle."""
        if button.get_active():
            self._current_tool = InpaintTool.RECT
            # Untoggle paint mask
            self._paint_mask_button.handler_block_by_func(self._on_paint_mask_toggled)
            self._paint_mask_button.set_active(False)
            self._paint_mask_button.handler_unblock_by_func(self._on_paint_mask_toggled)
        else:
            self._current_tool = InpaintTool.NONE
        if self._on_inpaint_tool_changed:
            self._on_inpaint_tool_changed(self._current_tool)

    def _on_paint_mask_toggled(self, button):
        """Handle Paint Mask tool toggle."""
        if button.get_active():
            self._current_tool = InpaintTool.PAINT
            # Untoggle rect mask
            self._rect_mask_button.handler_block_by_func(self._on_rect_mask_toggled)
            self._rect_mask_button.set_active(False)
            self._rect_mask_button.handler_unblock_by_func(self._on_rect_mask_toggled)
        else:
            self._current_tool = InpaintTool.NONE
        if self._on_inpaint_tool_changed:
            self._on_inpaint_tool_changed(self._current_tool)

    def _on_clear_masks_clicked(self, button):
        """Handle Clear Masks button click."""
        if self._on_clear_masks:
            self._on_clear_masks()

    def _on_generate_inpaint_clicked(self, button):
        """Handle Generate Inpaint button click."""
        if self._on_generate_inpaint:
            self._on_generate_inpaint()

    def _update_inpaint_ui(self):
        """Update visibility of inpaint-related buttons."""
        visible = self._inpaint_mode
        self._rect_mask_button.set_visible(visible)
        self._paint_mask_button.set_visible(visible)
        self._clear_masks_button.set_visible(visible)
        self._generate_inpaint_button.set_visible(visible)
        # Hide normal generation buttons in inpaint mode
        self._generate_button.set_visible(not visible)
        self._img2img_button.set_visible(not visible)

    def set_state(self, state: GenerationState):
        """Update toolbar state based on generation state."""
        if state == GenerationState.IDLE:
            self._load_button.set_sensitive(True)
            # Restore generation button sensitivity based on model loaded state
            self._update_generation_buttons_sensitivity()
            self._cancel_button.set_visible(False)
            self._progress_bar.set_visible(False)
            # Restore upscale button visibility
            self._upscale_button.set_visible(True)
            # Restore visibility based on inpaint mode
            if self._inpaint_mode:
                self._generate_button.set_visible(False)
                self._img2img_button.set_visible(False)
                self._rect_mask_button.set_visible(True)
                self._paint_mask_button.set_visible(True)
                self._clear_masks_button.set_visible(True)
                self._generate_inpaint_button.set_visible(True)
            else:
                self._generate_button.set_visible(True)
                self._img2img_button.set_visible(True)

        elif state == GenerationState.LOADING:
            self._load_button.set_sensitive(False)
            self._clear_button.set_sensitive(False)
            self._generate_button.set_sensitive(False)
            self._generate_button.set_visible(True)
            self._img2img_button.set_sensitive(False)
            self._img2img_button.set_visible(True)
            self._upscale_button.set_sensitive(False)
            self._cancel_button.set_visible(False)
            self._progress_bar.set_visible(True)

        elif state == GenerationState.GENERATING:
            self._load_button.set_sensitive(False)
            self._clear_button.set_sensitive(False)
            self._generate_button.set_visible(False)
            self._img2img_button.set_visible(False)
            self._upscale_button.set_visible(False)
            self._inpaint_toggle.set_sensitive(False)
            self._rect_mask_button.set_visible(False)
            self._paint_mask_button.set_visible(False)
            self._clear_masks_button.set_visible(False)
            self._generate_inpaint_button.set_visible(False)
            self._cancel_button.set_visible(True)
            self._progress_bar.set_visible(True)

        elif state == GenerationState.CANCELLING:
            self._load_button.set_sensitive(False)
            self._clear_button.set_sensitive(False)
            self._generate_button.set_visible(False)
            self._img2img_button.set_visible(False)
            self._upscale_button.set_visible(False)
            self._inpaint_toggle.set_sensitive(False)
            self._cancel_button.set_sensitive(False)
            self._progress_bar.set_visible(True)

    def set_model_loaded(self, loaded: bool):
        """Update state based on whether a model is loaded."""
        self._model_loaded = loaded
        self._update_generation_buttons_sensitivity()

    def _update_generation_buttons_sensitivity(self):
        """Update sensitivity of all generation-related buttons based on model loaded state."""
        loaded = self._model_loaded
        # Generate buttons are always active - they will auto-load models if needed
        self._generate_button.set_sensitive(True)
        self._img2img_button.set_sensitive(True)
        self._generate_inpaint_button.set_sensitive(True)
        # Clear and inpaint toggle only make sense when model is loaded
        self._clear_button.set_sensitive(loaded)
        self._inpaint_toggle.set_sensitive(loaded)

    def set_has_image(self, has_image: bool):
        """Update img2img button based on whether there's an image to use."""
        # img2img requires both a loaded model and an image
        # The model check is handled by set_model_loaded
        self._img2img_button.set_tooltip_text(
            "Generate a new image based on the current image" if has_image
            else "Load an image first to use Image to Image"
        )
        # Show/hide inpaint toggle based on image presence
        self._inpaint_separator.set_visible(has_image)
        self._inpaint_toggle.set_visible(has_image)
        # If no image and inpaint mode was on, turn it off
        if not has_image and self._inpaint_mode:
            self._inpaint_toggle.set_active(False)

    @property
    def inpaint_mode(self) -> bool:
        """Check if inpaint mode is active."""
        return self._inpaint_mode

    @property
    def current_tool(self) -> InpaintTool:
        """Get the current inpaint tool."""
        return self._current_tool

    def exit_inpaint_mode(self):
        """Exit inpaint mode programmatically."""
        if self._inpaint_mode:
            self._inpaint_toggle.set_active(False)

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

    def set_upscale_enabled(self, enabled: bool, has_image: bool = True):
        """Update upscale button sensitivity based on upscaling enabled state and image presence."""
        self._upscale_enabled = enabled
        # Upscale requires: upscaling enabled and an image present
        # (does not require the SD model to be loaded)
        self._upscale_button.set_sensitive(enabled and has_image)
