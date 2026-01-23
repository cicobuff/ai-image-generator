"""Main toolbar widget with action buttons."""

from typing import Optional, Callable
from enum import Enum

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.core.generation_service import GenerationState
from src.utils.constants import SIZE_PRESETS


class InpaintTool(Enum):
    """Inpaint mask drawing tools."""
    NONE = "none"
    RECT = "rect"
    PAINT = "paint"


class OutpaintTool(Enum):
    """Outpaint mask drawing tools."""
    NONE = "none"
    EDGE = "edge"  # Draw from edge outward


class CropTool(Enum):
    """Crop mask drawing tools."""
    NONE = "none"
    DRAW = "draw"  # Draw crop region


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
        on_outpaint_mode_changed: Optional[Callable[[bool], None]] = None,
        on_outpaint_tool_changed: Optional[Callable[[str], None]] = None,
        on_clear_outpaint_masks: Optional[Callable[[], None]] = None,
        on_generate_outpaint: Optional[Callable[[], None]] = None,
        on_crop_mode_changed: Optional[Callable[[bool], None]] = None,
        on_crop_tool_changed: Optional[Callable[["CropTool"], None]] = None,
        on_clear_crop_mask: Optional[Callable[[], None]] = None,
        on_crop_image: Optional[Callable[[], None]] = None,
        on_crop_size_changed: Optional[Callable[[int, int], None]] = None,
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
        self._on_outpaint_mode_changed = on_outpaint_mode_changed
        self._on_outpaint_tool_changed = on_outpaint_tool_changed
        self._on_clear_outpaint_masks = on_clear_outpaint_masks
        self._on_generate_outpaint = on_generate_outpaint
        self._on_crop_mode_changed = on_crop_mode_changed
        self._on_crop_tool_changed = on_crop_tool_changed
        self._on_clear_crop_mask = on_clear_crop_mask
        self._on_crop_image = on_crop_image
        self._on_crop_size_changed = on_crop_size_changed

        self._inpaint_mode = False
        self._outpaint_mode = False
        self._crop_mode = False
        self._current_tool = InpaintTool.NONE
        self._current_outpaint_tool = OutpaintTool.NONE
        self._current_crop_tool = CropTool.NONE
        self._has_crop_mask = False
        self._model_loaded = False
        self._upscale_enabled = False
        self._has_image = False

        self.add_css_class("toolbar")
        self._build_ui()
        self._update_generation_buttons_sensitivity()

    def _build_ui(self):
        """Build the toolbar UI."""
        # Load Models button with icon
        self._load_button = Gtk.Button()
        load_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        load_icon = Gtk.Image.new_from_icon_name("document-open-symbolic")
        load_label = Gtk.Label(label="Load Models")
        load_box.append(load_icon)
        load_box.append(load_label)
        self._load_button.set_child(load_box)
        self._load_button.add_css_class("dark-grey-button")
        self._load_button.connect("clicked", self._on_load_clicked)
        self.append(self._load_button)

        # Clear Models button with icon
        self._clear_button = Gtk.Button()
        clear_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        clear_icon = Gtk.Image.new_from_icon_name("edit-clear-all-symbolic")
        clear_label = Gtk.Label(label="Clear Models")
        clear_box.append(clear_icon)
        clear_box.append(clear_label)
        self._clear_button.set_child(clear_box)
        self._clear_button.add_css_class("dark-grey-button")
        self._clear_button.connect("clicked", self._on_clear_clicked)
        self.append(self._clear_button)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        separator.set_margin_start(8)
        separator.set_margin_end(8)
        self.append(separator)

        # Generate button with icon
        self._generate_button = Gtk.Button()
        generate_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        generate_icon = Gtk.Image.new_from_icon_name("media-playback-start-symbolic")
        generate_label = Gtk.Label(label="Generate")
        generate_box.append(generate_icon)
        generate_box.append(generate_label)
        self._generate_button.set_child(generate_box)
        self._generate_button.add_css_class("blue-button")
        self._generate_button.connect("clicked", self._on_generate_clicked)
        self.append(self._generate_button)

        # Image to Image button with icon
        self._img2img_button = Gtk.Button()
        img2img_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        img2img_icon = Gtk.Image.new_from_icon_name("image-x-generic-symbolic")
        img2img_label = Gtk.Label(label="Img2Img")
        img2img_box.append(img2img_icon)
        img2img_box.append(img2img_label)
        self._img2img_button.set_child(img2img_box)
        self._img2img_button.add_css_class("blue-button")
        self._img2img_button.set_tooltip_text("Generate a new image based on the current image")
        self._img2img_button.connect("clicked", self._on_img2img_clicked)
        self.append(self._img2img_button)

        # Upscale button with icon
        self._upscale_button = Gtk.Button()
        upscale_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        upscale_icon = Gtk.Image.new_from_icon_name("zoom-in-symbolic")
        upscale_label = Gtk.Label(label="Upscale")
        upscale_box.append(upscale_icon)
        upscale_box.append(upscale_label)
        self._upscale_button.set_child(upscale_box)
        self._upscale_button.add_css_class("blue-button")
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

        # Inpaint Mode toggle with icon
        self._inpaint_toggle = Gtk.ToggleButton()
        inpaint_mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        inpaint_mode_icon = Gtk.Image.new_from_icon_name("applications-graphics-symbolic")
        inpaint_mode_label = Gtk.Label(label="Inpaint Mode")
        inpaint_mode_box.append(inpaint_mode_icon)
        inpaint_mode_box.append(inpaint_mode_label)
        self._inpaint_toggle.set_child(inpaint_mode_box)
        self._inpaint_toggle.set_tooltip_text("Enable inpainting mode to draw masks")
        self._inpaint_toggle.add_css_class("inpaint-toggle")
        self._inpaint_toggle.connect("toggled", self._on_inpaint_toggled)
        self._inpaint_toggle.set_visible(False)
        self.append(self._inpaint_toggle)

        # Rect Mask tool button with icon (visible in inpaint mode)
        self._rect_mask_button = Gtk.ToggleButton()
        rect_mask_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        rect_mask_icon = Gtk.Image.new_from_icon_name("view-fullscreen-symbolic")
        rect_mask_label = Gtk.Label(label="Rect Mask")
        rect_mask_box.append(rect_mask_icon)
        rect_mask_box.append(rect_mask_label)
        self._rect_mask_button.set_child(rect_mask_box)
        self._rect_mask_button.set_tooltip_text("Draw rectangular masks")
        self._rect_mask_button.add_css_class("green-toggle")
        self._rect_mask_button.connect("toggled", self._on_rect_mask_toggled)
        self._rect_mask_button.set_visible(False)
        self.append(self._rect_mask_button)

        # Paint Mask tool button with icon (visible in inpaint mode)
        self._paint_mask_button = Gtk.ToggleButton()
        paint_mask_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        paint_mask_icon = Gtk.Image.new_from_icon_name("radio-symbolic")
        paint_mask_label = Gtk.Label(label="Paint Mask")
        paint_mask_box.append(paint_mask_icon)
        paint_mask_box.append(paint_mask_label)
        self._paint_mask_button.set_child(paint_mask_box)
        self._paint_mask_button.set_tooltip_text("Paint masks with brush (25px radius)")
        self._paint_mask_button.add_css_class("green-toggle")
        self._paint_mask_button.connect("toggled", self._on_paint_mask_toggled)
        self._paint_mask_button.set_visible(False)
        self.append(self._paint_mask_button)

        # Clear Masks button with icon (visible in inpaint mode)
        self._clear_masks_button = Gtk.Button()
        clear_masks_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        clear_masks_icon = Gtk.Image.new_from_icon_name("edit-clear-symbolic")
        clear_masks_label = Gtk.Label(label="Clear Masks")
        clear_masks_box.append(clear_masks_icon)
        clear_masks_box.append(clear_masks_label)
        self._clear_masks_button.set_child(clear_masks_box)
        self._clear_masks_button.set_tooltip_text("Clear all drawn masks")
        self._clear_masks_button.add_css_class("green-button")
        self._clear_masks_button.connect("clicked", self._on_clear_masks_clicked)
        self._clear_masks_button.set_visible(False)
        self.append(self._clear_masks_button)

        # Generate Inpaint button with icon (visible in inpaint mode)
        self._generate_inpaint_button = Gtk.Button()
        inpaint_gen_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        inpaint_gen_icon = Gtk.Image.new_from_icon_name("media-playback-start-symbolic")
        inpaint_gen_label = Gtk.Label(label="Generate Inpaint")
        inpaint_gen_box.append(inpaint_gen_icon)
        inpaint_gen_box.append(inpaint_gen_label)
        self._generate_inpaint_button.set_child(inpaint_gen_box)
        self._generate_inpaint_button.add_css_class("green-button")
        self._generate_inpaint_button.set_tooltip_text("Generate inpainted image in masked areas")
        self._generate_inpaint_button.connect("clicked", self._on_generate_inpaint_clicked)
        self._generate_inpaint_button.set_visible(False)
        self.append(self._generate_inpaint_button)

        # Separator before outpaint
        self._outpaint_separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self._outpaint_separator.set_margin_start(8)
        self._outpaint_separator.set_margin_end(8)
        self._outpaint_separator.set_visible(False)
        self.append(self._outpaint_separator)

        # Outpaint Mode toggle with icon
        self._outpaint_toggle = Gtk.ToggleButton()
        outpaint_mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        outpaint_mode_icon = Gtk.Image.new_from_icon_name("view-fullscreen-symbolic")
        outpaint_mode_label = Gtk.Label(label="Outpaint Mode")
        outpaint_mode_box.append(outpaint_mode_icon)
        outpaint_mode_box.append(outpaint_mode_label)
        self._outpaint_toggle.set_child(outpaint_mode_box)
        self._outpaint_toggle.set_tooltip_text("Enable outpainting mode to extend image edges")
        self._outpaint_toggle.add_css_class("outpaint-toggle")
        self._outpaint_toggle.connect("toggled", self._on_outpaint_toggled)
        self._outpaint_toggle.set_visible(False)
        self.append(self._outpaint_toggle)

        # Outpaint Mask tool button with icon (visible in outpaint mode)
        self._outpaint_mask_button = Gtk.ToggleButton()
        outpaint_mask_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        outpaint_mask_icon = Gtk.Image.new_from_icon_name("object-flip-horizontal-symbolic")
        outpaint_mask_label = Gtk.Label(label="Edge Mask")
        outpaint_mask_box.append(outpaint_mask_icon)
        outpaint_mask_box.append(outpaint_mask_label)
        self._outpaint_mask_button.set_child(outpaint_mask_box)
        self._outpaint_mask_button.set_tooltip_text("Draw outpaint masks from image edges")
        self._outpaint_mask_button.add_css_class("pink-toggle")
        self._outpaint_mask_button.connect("toggled", self._on_outpaint_mask_toggled)
        self._outpaint_mask_button.set_visible(False)
        self.append(self._outpaint_mask_button)

        # Clear Outpaint Masks button with icon (visible in outpaint mode)
        self._clear_outpaint_masks_button = Gtk.Button()
        clear_outpaint_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        clear_outpaint_icon = Gtk.Image.new_from_icon_name("edit-clear-symbolic")
        clear_outpaint_label = Gtk.Label(label="Clear Masks")
        clear_outpaint_box.append(clear_outpaint_icon)
        clear_outpaint_box.append(clear_outpaint_label)
        self._clear_outpaint_masks_button.set_child(clear_outpaint_box)
        self._clear_outpaint_masks_button.set_tooltip_text("Clear all outpaint masks")
        self._clear_outpaint_masks_button.add_css_class("pink-button")
        self._clear_outpaint_masks_button.connect("clicked", self._on_clear_outpaint_masks_clicked)
        self._clear_outpaint_masks_button.set_visible(False)
        self.append(self._clear_outpaint_masks_button)

        # Generate Outpaint button with icon (visible in outpaint mode)
        self._generate_outpaint_button = Gtk.Button()
        outpaint_gen_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        outpaint_gen_icon = Gtk.Image.new_from_icon_name("media-playback-start-symbolic")
        outpaint_gen_label = Gtk.Label(label="Generate Outpaint")
        outpaint_gen_box.append(outpaint_gen_icon)
        outpaint_gen_box.append(outpaint_gen_label)
        self._generate_outpaint_button.set_child(outpaint_gen_box)
        self._generate_outpaint_button.add_css_class("pink-button")
        self._generate_outpaint_button.set_tooltip_text("Generate outpainted image extending masked edges")
        self._generate_outpaint_button.connect("clicked", self._on_generate_outpaint_clicked)
        self._generate_outpaint_button.set_visible(False)
        self.append(self._generate_outpaint_button)

        # Separator before crop
        self._crop_separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self._crop_separator.set_margin_start(8)
        self._crop_separator.set_margin_end(8)
        self._crop_separator.set_visible(False)
        self.append(self._crop_separator)

        # Crop Mode toggle with icon
        self._crop_toggle = Gtk.ToggleButton()
        crop_mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        crop_mode_icon = Gtk.Image.new_from_icon_name("edit-cut-symbolic")
        crop_mode_label = Gtk.Label(label="Crop Mode")
        crop_mode_box.append(crop_mode_icon)
        crop_mode_box.append(crop_mode_label)
        self._crop_toggle.set_child(crop_mode_box)
        self._crop_toggle.set_tooltip_text("Enable crop mode to select a region to crop")
        self._crop_toggle.add_css_class("crop-toggle")
        self._crop_toggle.connect("toggled", self._on_crop_toggled)
        self._crop_toggle.set_visible(False)
        self.append(self._crop_toggle)

        # Crop Mask tool button with icon (visible in crop mode)
        self._crop_mask_button = Gtk.ToggleButton()
        crop_mask_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        crop_mask_icon = Gtk.Image.new_from_icon_name("view-fullscreen-symbolic")
        crop_mask_label = Gtk.Label(label="Crop Mask")
        crop_mask_box.append(crop_mask_icon)
        crop_mask_box.append(crop_mask_label)
        self._crop_mask_button.set_child(crop_mask_box)
        self._crop_mask_button.set_tooltip_text("Draw crop region on the image")
        self._crop_mask_button.add_css_class("purple-toggle")
        self._crop_mask_button.connect("toggled", self._on_crop_mask_toggled)
        self._crop_mask_button.set_visible(False)
        self.append(self._crop_mask_button)

        # Clear Crop Mask button with icon (visible in crop mode)
        self._clear_crop_mask_button = Gtk.Button()
        clear_crop_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        clear_crop_icon = Gtk.Image.new_from_icon_name("edit-clear-symbolic")
        clear_crop_label = Gtk.Label(label="Clear Mask")
        clear_crop_box.append(clear_crop_icon)
        clear_crop_box.append(clear_crop_label)
        self._clear_crop_mask_button.set_child(clear_crop_box)
        self._clear_crop_mask_button.set_tooltip_text("Clear crop mask")
        self._clear_crop_mask_button.add_css_class("purple-button")
        self._clear_crop_mask_button.connect("clicked", self._on_clear_crop_mask_clicked)
        self._clear_crop_mask_button.set_visible(False)
        self.append(self._clear_crop_mask_button)

        # Crop Image button with icon (visible in crop mode)
        self._crop_image_button = Gtk.Button()
        crop_img_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        crop_img_icon = Gtk.Image.new_from_icon_name("object-select-symbolic")
        crop_img_label = Gtk.Label(label="Crop Image")
        crop_img_box.append(crop_img_icon)
        crop_img_box.append(crop_img_label)
        self._crop_image_button.set_child(crop_img_box)
        self._crop_image_button.add_css_class("purple-button")
        self._crop_image_button.set_tooltip_text("Crop image to the selected region")
        self._crop_image_button.connect("clicked", self._on_crop_image_clicked)
        self._crop_image_button.set_visible(False)
        self.append(self._crop_image_button)

        # Crop Size selector dropdown (visible in crop mode)
        self._crop_size_dropdown = Gtk.DropDown()
        size_labels = ["Select Size"] + list(SIZE_PRESETS.keys())
        string_list = Gtk.StringList.new(size_labels)
        self._crop_size_dropdown.set_model(string_list)
        self._crop_size_dropdown.set_selected(0)  # "Select Size"
        self._crop_size_dropdown.add_css_class("purple-button")
        self._crop_size_dropdown.set_tooltip_text("Set crop mask to a preset size")
        self._crop_size_dropdown.connect("notify::selected", self._on_crop_size_selected)
        self._crop_size_dropdown.set_visible(False)
        self.append(self._crop_size_dropdown)

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

        # Turn off other modes when this mode is activated
        if self._inpaint_mode:
            if self._outpaint_mode:
                self._outpaint_toggle.handler_block_by_func(self._on_outpaint_toggled)
                self._outpaint_toggle.set_active(False)
                self._outpaint_toggle.handler_unblock_by_func(self._on_outpaint_toggled)
                self._outpaint_mode = False
                self._current_outpaint_tool = OutpaintTool.NONE
                self._outpaint_mask_button.set_active(False)
                self._update_outpaint_ui()
                if self._on_outpaint_mode_changed:
                    self._on_outpaint_mode_changed(False)
            if self._crop_mode:
                self._crop_toggle.handler_block_by_func(self._on_crop_toggled)
                self._crop_toggle.set_active(False)
                self._crop_toggle.handler_unblock_by_func(self._on_crop_toggled)
                self._crop_mode = False
                self._current_crop_tool = CropTool.NONE
                self._crop_mask_button.set_active(False)
                self._update_crop_ui()
                if self._on_crop_mode_changed:
                    self._on_crop_mode_changed(False)

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

    def _on_outpaint_toggled(self, button):
        """Handle Outpaint Mode toggle."""
        self._outpaint_mode = button.get_active()

        # Turn off other modes when this mode is activated
        if self._outpaint_mode:
            if self._inpaint_mode:
                self._inpaint_toggle.handler_block_by_func(self._on_inpaint_toggled)
                self._inpaint_toggle.set_active(False)
                self._inpaint_toggle.handler_unblock_by_func(self._on_inpaint_toggled)
                self._inpaint_mode = False
                self._current_tool = InpaintTool.NONE
                self._rect_mask_button.set_active(False)
                self._paint_mask_button.set_active(False)
                self._update_inpaint_ui()
                if self._on_inpaint_mode_changed:
                    self._on_inpaint_mode_changed(False)
            if self._crop_mode:
                self._crop_toggle.handler_block_by_func(self._on_crop_toggled)
                self._crop_toggle.set_active(False)
                self._crop_toggle.handler_unblock_by_func(self._on_crop_toggled)
                self._crop_mode = False
                self._current_crop_tool = CropTool.NONE
                self._crop_mask_button.set_active(False)
                self._update_crop_ui()
                if self._on_crop_mode_changed:
                    self._on_crop_mode_changed(False)

        self._update_outpaint_ui()
        if self._on_outpaint_mode_changed:
            self._on_outpaint_mode_changed(self._outpaint_mode)
        # Reset tool when exiting outpaint mode
        if not self._outpaint_mode:
            self._current_outpaint_tool = OutpaintTool.NONE
            self._outpaint_mask_button.set_active(False)
            if self._on_outpaint_tool_changed:
                self._on_outpaint_tool_changed(OutpaintTool.NONE)

    def _on_outpaint_mask_toggled(self, button):
        """Handle Outpaint Mask tool toggle."""
        if button.get_active():
            self._current_outpaint_tool = OutpaintTool.EDGE
        else:
            self._current_outpaint_tool = OutpaintTool.NONE
        if self._on_outpaint_tool_changed:
            self._on_outpaint_tool_changed(self._current_outpaint_tool)

    def _on_clear_outpaint_masks_clicked(self, button):
        """Handle Clear Outpaint Masks button click."""
        if self._on_clear_outpaint_masks:
            self._on_clear_outpaint_masks()

    def _on_generate_outpaint_clicked(self, button):
        """Handle Generate Outpaint button click."""
        if self._on_generate_outpaint:
            self._on_generate_outpaint()

    def _on_crop_toggled(self, button):
        """Handle Crop Mode toggle."""
        self._crop_mode = button.get_active()

        # Turn off other modes when this mode is activated
        if self._crop_mode:
            if self._inpaint_mode:
                self._inpaint_toggle.handler_block_by_func(self._on_inpaint_toggled)
                self._inpaint_toggle.set_active(False)
                self._inpaint_toggle.handler_unblock_by_func(self._on_inpaint_toggled)
                self._inpaint_mode = False
                self._current_tool = InpaintTool.NONE
                self._rect_mask_button.set_active(False)
                self._paint_mask_button.set_active(False)
                self._update_inpaint_ui()
                if self._on_inpaint_mode_changed:
                    self._on_inpaint_mode_changed(False)
            if self._outpaint_mode:
                self._outpaint_toggle.handler_block_by_func(self._on_outpaint_toggled)
                self._outpaint_toggle.set_active(False)
                self._outpaint_toggle.handler_unblock_by_func(self._on_outpaint_toggled)
                self._outpaint_mode = False
                self._current_outpaint_tool = OutpaintTool.NONE
                self._outpaint_mask_button.set_active(False)
                self._update_outpaint_ui()
                if self._on_outpaint_mode_changed:
                    self._on_outpaint_mode_changed(False)

        self._update_crop_ui()
        if self._on_crop_mode_changed:
            self._on_crop_mode_changed(self._crop_mode)
        # Reset tool when exiting crop mode
        if not self._crop_mode:
            self._current_crop_tool = CropTool.NONE
            self._crop_mask_button.set_active(False)
            if self._on_crop_tool_changed:
                self._on_crop_tool_changed(CropTool.NONE)

    def _on_crop_mask_toggled(self, button):
        """Handle Crop Mask tool toggle."""
        if button.get_active():
            self._current_crop_tool = CropTool.DRAW
        else:
            self._current_crop_tool = CropTool.NONE
        if self._on_crop_tool_changed:
            self._on_crop_tool_changed(self._current_crop_tool)

    def _on_clear_crop_mask_clicked(self, button):
        """Handle Clear Crop Mask button click."""
        if self._on_clear_crop_mask:
            self._on_clear_crop_mask()
        # Re-enable crop mask button after clearing
        self._has_crop_mask = False
        self._crop_mask_button.set_sensitive(True)

    def _on_crop_image_clicked(self, button):
        """Handle Crop Image button click."""
        if self._on_crop_image:
            self._on_crop_image()

    def _on_crop_size_selected(self, dropdown, param):
        """Handle crop size selection from dropdown."""
        selected = dropdown.get_selected()
        if selected == 0:
            # "Select Size" placeholder - do nothing
            return

        # Get the size key (offset by 1 due to placeholder)
        size_keys = list(SIZE_PRESETS.keys())
        if selected <= len(size_keys):
            size_key = size_keys[selected - 1]
            width, height = SIZE_PRESETS[size_key]
            if self._on_crop_size_changed:
                self._on_crop_size_changed(width, height)

            # Activate the crop mask tool so the mask can be moved/resized
            if not self._crop_mask_button.get_active():
                self._crop_mask_button.set_active(True)

        # Reset dropdown to placeholder after selection
        dropdown.set_selected(0)

    def _update_crop_ui(self):
        """Update visibility of crop-related buttons."""
        visible = self._crop_mode
        self._crop_mask_button.set_visible(visible)
        self._clear_crop_mask_button.set_visible(visible)
        self._crop_image_button.set_visible(visible)
        self._crop_size_dropdown.set_visible(visible)
        # Disable normal generation buttons in crop mode (keep visible)
        self._generate_button.set_sensitive(not visible)
        self._img2img_button.set_sensitive(not visible)

    def set_crop_mask_exists(self, exists: bool):
        """Update crop mask button sensitivity based on whether a mask exists."""
        self._has_crop_mask = exists
        self._crop_mask_button.set_sensitive(not exists)

    def _update_outpaint_ui(self):
        """Update visibility of outpaint-related buttons."""
        visible = self._outpaint_mode
        self._outpaint_mask_button.set_visible(visible)
        self._clear_outpaint_masks_button.set_visible(visible)
        self._generate_outpaint_button.set_visible(visible)
        # Disable normal generation buttons in outpaint mode (keep visible)
        self._generate_button.set_sensitive(not visible)
        self._img2img_button.set_sensitive(not visible)

    def _update_inpaint_ui(self):
        """Update visibility of inpaint-related buttons."""
        visible = self._inpaint_mode
        self._rect_mask_button.set_visible(visible)
        self._paint_mask_button.set_visible(visible)
        self._clear_masks_button.set_visible(visible)
        self._generate_inpaint_button.set_visible(visible)
        # Disable normal generation buttons in inpaint mode (keep visible)
        self._generate_button.set_sensitive(not visible)
        self._img2img_button.set_sensitive(not visible)

    def set_state(self, state: GenerationState):
        """Update toolbar state based on generation state."""
        if state == GenerationState.IDLE:
            self._load_button.set_sensitive(True)
            self._clear_button.set_sensitive(self._model_loaded)
            # Restore generation button sensitivity
            self._update_generation_buttons_sensitivity()
            self._cancel_button.set_visible(False)
            self._progress_bar.set_visible(False)
            # Restore inpaint controls sensitivity (based on image presence, not model loaded)
            self._inpaint_toggle.set_sensitive(self._has_image)
            self._rect_mask_button.set_sensitive(True)
            self._paint_mask_button.set_sensitive(True)
            self._clear_masks_button.set_sensitive(True)
            # Restore outpaint controls sensitivity
            self._outpaint_toggle.set_sensitive(self._has_image)
            self._outpaint_mask_button.set_sensitive(True)
            self._clear_outpaint_masks_button.set_sensitive(True)
            # Restore crop controls sensitivity
            self._crop_toggle.set_sensitive(self._has_image)
            self._crop_mask_button.set_sensitive(not self._has_crop_mask)
            self._clear_crop_mask_button.set_sensitive(True)
            self._crop_image_button.set_sensitive(True)
            # Restore state based on inpaint/outpaint/crop mode
            if self._inpaint_mode:
                # Keep Generate/Img2Img visible but disabled in inpaint mode
                self._generate_button.set_sensitive(False)
                self._img2img_button.set_sensitive(False)
                self._rect_mask_button.set_visible(True)
                self._paint_mask_button.set_visible(True)
                self._clear_masks_button.set_visible(True)
                self._generate_inpaint_button.set_visible(True)
            elif self._outpaint_mode:
                # Keep Generate/Img2Img visible but disabled in outpaint mode
                self._generate_button.set_sensitive(False)
                self._img2img_button.set_sensitive(False)
                self._outpaint_mask_button.set_visible(True)
                self._clear_outpaint_masks_button.set_visible(True)
                self._generate_outpaint_button.set_visible(True)
            elif self._crop_mode:
                # Keep Generate/Img2Img visible but disabled in crop mode
                self._generate_button.set_sensitive(False)
                self._img2img_button.set_sensitive(False)
                self._crop_mask_button.set_visible(True)
                self._clear_crop_mask_button.set_visible(True)
                self._crop_image_button.set_visible(True)
            else:
                # Restore sensitivity when not in edit mode
                self._generate_button.set_sensitive(True)
                self._img2img_button.set_sensitive(True)

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
            self._generate_button.set_sensitive(False)
            self._img2img_button.set_sensitive(False)
            self._upscale_button.set_sensitive(False)
            self._inpaint_toggle.set_sensitive(False)
            self._rect_mask_button.set_sensitive(False)
            self._paint_mask_button.set_sensitive(False)
            self._clear_masks_button.set_sensitive(False)
            self._generate_inpaint_button.set_sensitive(False)
            self._outpaint_toggle.set_sensitive(False)
            self._outpaint_mask_button.set_sensitive(False)
            self._clear_outpaint_masks_button.set_sensitive(False)
            self._generate_outpaint_button.set_sensitive(False)
            self._crop_toggle.set_sensitive(False)
            self._crop_mask_button.set_sensitive(False)
            self._clear_crop_mask_button.set_sensitive(False)
            self._crop_image_button.set_sensitive(False)
            self._cancel_button.set_visible(True)
            self._progress_bar.set_visible(True)

        elif state == GenerationState.CANCELLING:
            self._load_button.set_sensitive(False)
            self._clear_button.set_sensitive(False)
            self._generate_button.set_sensitive(False)
            self._img2img_button.set_sensitive(False)
            self._upscale_button.set_sensitive(False)
            self._inpaint_toggle.set_sensitive(False)
            self._outpaint_toggle.set_sensitive(False)
            self._crop_toggle.set_sensitive(False)
            self._cancel_button.set_sensitive(False)
            self._progress_bar.set_visible(True)

    def set_model_loaded(self, loaded: bool):
        """Update state based on whether a model is loaded."""
        self._model_loaded = loaded
        self._update_generation_buttons_sensitivity()

    def _update_generation_buttons_sensitivity(self):
        """Update sensitivity of all generation-related buttons based on model loaded state and edit modes."""
        loaded = self._model_loaded
        # Generate buttons are always active - they will auto-load models if needed
        # But they should be disabled in inpaint/outpaint/crop mode (keep visible though)
        in_edit_mode = self._inpaint_mode or self._outpaint_mode or self._crop_mode
        self._generate_button.set_sensitive(not in_edit_mode)
        self._img2img_button.set_sensitive(not in_edit_mode)
        self._generate_inpaint_button.set_sensitive(True)
        self._generate_outpaint_button.set_sensitive(True)
        # Clear only makes sense when model is loaded
        self._clear_button.set_sensitive(loaded)
        # Edit toggles are based on image presence, not model loaded (handled in set_has_image)

    def set_has_image(self, has_image: bool):
        """Update img2img button based on whether there's an image to use."""
        self._has_image = has_image
        # img2img requires both a loaded model and an image
        # The model check is handled by set_model_loaded
        self._img2img_button.set_tooltip_text(
            "Generate a new image based on the current image" if has_image
            else "Load an image first to use Image to Image"
        )
        # Show/hide inpaint toggle based on image presence
        self._inpaint_separator.set_visible(has_image)
        self._inpaint_toggle.set_visible(has_image)
        # Inpaint toggle is active when there's an image (doesn't require model to be loaded)
        self._inpaint_toggle.set_sensitive(has_image)
        # If no image and inpaint mode was on, turn it off
        if not has_image and self._inpaint_mode:
            self._inpaint_toggle.set_active(False)

        # Show/hide outpaint toggle based on image presence
        self._outpaint_separator.set_visible(has_image)
        self._outpaint_toggle.set_visible(has_image)
        self._outpaint_toggle.set_sensitive(has_image)
        # If no image and outpaint mode was on, turn it off
        if not has_image and self._outpaint_mode:
            self._outpaint_toggle.set_active(False)

        # Show/hide crop toggle based on image presence
        self._crop_separator.set_visible(has_image)
        self._crop_toggle.set_visible(has_image)
        self._crop_toggle.set_sensitive(has_image)
        # If no image and crop mode was on, turn it off
        if not has_image and self._crop_mode:
            self._crop_toggle.set_active(False)

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

    @property
    def outpaint_mode(self) -> bool:
        """Check if outpaint mode is active."""
        return self._outpaint_mode

    @property
    def current_outpaint_tool(self) -> OutpaintTool:
        """Get the current outpaint tool."""
        return self._current_outpaint_tool

    def exit_outpaint_mode(self):
        """Exit outpaint mode programmatically."""
        if self._outpaint_mode:
            self._outpaint_toggle.set_active(False)

    @property
    def crop_mode(self) -> bool:
        """Check if crop mode is active."""
        return self._crop_mode

    @property
    def current_crop_tool(self) -> CropTool:
        """Get the current crop tool."""
        return self._current_crop_tool

    def exit_crop_mode(self):
        """Exit crop mode programmatically."""
        if self._crop_mode:
            self._crop_toggle.set_active(False)

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
