"""Main work screen with 3-panel layout."""

from pathlib import Path
import threading

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from src.core.config import config_manager
from src.core.model_manager import model_manager, ModelType
from src.core.generation_service import generation_service, GenerationState, GenerationResult
from src.ui.widgets.model_selector import ModelSelector
from src.ui.widgets.vram_display import VRAMDisplay
from src.ui.widgets.generation_params import GenerationParamsWidget
from src.ui.widgets.prompt_entry import PromptPanel
from src.ui.widgets.image_display import ImageDisplayFrame
from src.ui.widgets.thumbnail_gallery import ThumbnailGallery
from src.ui.widgets.toolbar import Toolbar
from src.ui.widgets.upscale_settings import UpscaleSettingsWidget
from src.ui.widgets.toolbar import InpaintTool
from src.ui.widgets.image_display import MaskTool
from src.utils.metadata import load_metadata_from_image


class WorkScreen(Gtk.Box):
    """Main work screen with 3-panel layout for image generation."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._last_seed_was_random = False
        self._build_ui()
        self._connect_signals()
        self._initial_setup()

    def _build_ui(self):
        """Build the work screen UI."""
        # Toolbar
        self._toolbar = Toolbar(
            on_load=self._on_load_models,
            on_clear=self._on_clear_models,
            on_generate=self._on_generate,
            on_img2img=self._on_img2img,
            on_cancel=self._on_cancel,
            on_inpaint_mode_changed=self._on_inpaint_mode_changed,
            on_inpaint_tool_changed=self._on_inpaint_tool_changed,
            on_clear_masks=self._on_clear_masks,
            on_generate_inpaint=self._on_generate_inpaint,
        )
        self.append(self._toolbar)

        # Main content area with 3 panels
        self._paned_outer = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self._paned_outer.set_vexpand(True)
        self.append(self._paned_outer)

        # Left panel
        left_panel = self._create_left_panel()
        self._paned_outer.set_start_child(left_panel)
        self._paned_outer.set_resize_start_child(False)
        self._paned_outer.set_shrink_start_child(True)  # Allow shrinking

        # Right paned (center + right)
        self._paned_inner = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self._paned_outer.set_end_child(self._paned_inner)

        # Center panel
        center_panel = self._create_center_panel()
        self._paned_inner.set_start_child(center_panel)
        self._paned_inner.set_resize_start_child(True)
        self._paned_inner.set_shrink_start_child(True)  # Allow shrinking

        # Right panel
        right_panel = self._create_right_panel()
        self._paned_inner.set_end_child(right_panel)
        self._paned_inner.set_resize_end_child(False)
        self._paned_inner.set_shrink_end_child(True)  # Allow shrinking

        # Set initial sizes
        self._paned_outer.set_position(280)
        self._paned_inner.set_position(800)

        # Status bar
        self._status_bar = Gtk.Label(label="Ready")
        self._status_bar.add_css_class("status-bar")
        self._status_bar.set_halign(Gtk.Align.START)
        self._status_bar.set_margin_start(12)
        self._status_bar.set_margin_end(12)
        self._status_bar.set_margin_top(4)
        self._status_bar.set_margin_bottom(4)
        self.append(self._status_bar)

    def _create_left_panel(self) -> Gtk.Widget:
        """Create the left panel with model selectors and parameters."""
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.add_css_class("left-panel")
        scrolled.set_size_request(200, -1)  # Minimum width

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        scrolled.set_child(box)

        # Model selectors section
        model_header = Gtk.Label(label="Models")
        model_header.add_css_class("section-header")
        model_header.set_halign(Gtk.Align.START)
        box.append(model_header)

        # Checkpoint selector
        self._checkpoint_selector = ModelSelector(
            label="Checkpoint:",
            model_type=ModelType.CHECKPOINT,
        )
        box.append(self._checkpoint_selector)

        # VAE selector
        self._vae_selector = ModelSelector(
            label="VAE (optional):",
            model_type=ModelType.VAE,
        )
        box.append(self._vae_selector)

        # CLIP selector
        self._clip_selector = ModelSelector(
            label="CLIP (optional):",
            model_type=ModelType.CLIP,
        )
        box.append(self._clip_selector)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(8)
        separator.set_margin_bottom(8)
        box.append(separator)

        # VRAM display
        self._vram_display = VRAMDisplay()
        box.append(self._vram_display)

        # Separator
        separator2 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator2.set_margin_top(8)
        separator2.set_margin_bottom(8)
        box.append(separator2)

        # Generation parameters
        self._params_widget = GenerationParamsWidget()
        box.append(self._params_widget)

        # Separator
        separator3 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator3.set_margin_top(8)
        separator3.set_margin_bottom(8)
        box.append(separator3)

        # Upscale settings
        self._upscale_widget = UpscaleSettingsWidget()
        box.append(self._upscale_widget)

        return scrolled

    def _create_center_panel(self) -> Gtk.Widget:
        """Create the center panel with image display and prompts."""
        # Use vertical paned to allow resizing between image and prompts
        self._center_paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self._center_paned.add_css_class("center-panel")
        self._center_paned.set_margin_top(12)
        self._center_paned.set_margin_bottom(12)
        self._center_paned.set_margin_start(12)
        self._center_paned.set_margin_end(12)

        # Image display (top)
        self._image_display = ImageDisplayFrame()
        self._image_display.set_vexpand(True)
        self._image_display.set_size_request(-1, 200)  # Minimum height
        self._center_paned.set_start_child(self._image_display)
        self._center_paned.set_resize_start_child(True)
        self._center_paned.set_shrink_start_child(True)

        # Prompts (bottom)
        self._prompt_panel = PromptPanel()
        self._prompt_panel.set_size_request(-1, 100)  # Minimum height
        self._center_paned.set_end_child(self._prompt_panel)
        self._center_paned.set_resize_end_child(True)
        self._center_paned.set_shrink_end_child(True)

        # Set initial position (leave room for prompts)
        self._center_paned.set_position(500)

        return self._center_paned

    def _create_right_panel(self) -> Gtk.Widget:
        """Create the right panel with thumbnail gallery."""
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.add_css_class("right-panel")
        scrolled.set_size_request(150, -1)  # Minimum width

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        scrolled.set_child(box)

        # Thumbnail gallery
        self._thumbnail_gallery = ThumbnailGallery(
            on_image_selected=self._on_thumbnail_selected,
        )
        self._thumbnail_gallery.set_vexpand(True)
        box.append(self._thumbnail_gallery)

        return scrolled

    def _connect_signals(self):
        """Connect to generation service signals."""
        generation_service.add_state_changed_callback(self._on_state_changed)
        generation_service.add_progress_callback(self._on_progress)
        generation_service.add_step_progress_callback(self._on_step_progress)
        generation_service.add_generation_complete_callback(self._on_generation_complete)

    def _initial_setup(self):
        """Perform initial setup tasks."""
        # Scan for models in background
        self._scan_models()

        # Load existing thumbnails
        output_path = config_manager.config.get_output_path()
        self._thumbnail_gallery.load_from_directory(output_path)

        # Update toolbar state
        self._toolbar.set_model_loaded(generation_service.is_model_loaded)

    def _scan_models(self):
        """Scan for models in background thread."""
        def scan_thread():
            model_manager.scan_models(
                progress_callback=lambda msg: GLib.idle_add(
                    self._status_bar.set_text, msg
                )
            )
            GLib.idle_add(self._on_models_scanned)

        thread = threading.Thread(target=scan_thread, daemon=True)
        thread.start()

    def _on_models_scanned(self):
        """Called when model scanning is complete."""
        self._status_bar.set_text("Ready")

    def _on_load_models(self):
        """Handle Load Models button click."""
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
            return

        generation_service.load_models()

    def _on_clear_models(self):
        """Handle Clear Models button click."""
        generation_service.unload_models()
        self._toolbar.set_model_loaded(False)
        self._status_bar.set_text("Models unloaded")

    def _on_generate(self):
        """Handle Generate button click."""
        if not generation_service.is_model_loaded:
            self._status_bar.set_text("Please load models first")
            return

        positive = self._prompt_panel.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        negative = self._prompt_panel.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

        # Get upscale settings
        upscale_enabled = self._upscale_widget.is_enabled
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        generation_service.generate(
            params,
            upscale_enabled=upscale_enabled,
            upscale_model_path=upscale_model_path,
            upscale_model_name=upscale_model_name,
        )

    def _on_img2img(self):
        """Handle Image to Image button click."""
        if not generation_service.is_model_loaded:
            self._status_bar.set_text("Please load models first")
            return

        # Check if there's an image to use as input
        input_image = self._image_display.get_pil_image()
        if input_image is None:
            self._status_bar.set_text("Please load or generate an image first")
            return

        positive = self._prompt_panel.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        negative = self._prompt_panel.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)
        strength = self._params_widget.get_strength()

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

        # Get upscale settings
        upscale_enabled = self._upscale_widget.is_enabled
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        generation_service.generate_img2img(
            params,
            input_image=input_image,
            strength=strength,
            upscale_enabled=upscale_enabled,
            upscale_model_path=upscale_model_path,
            upscale_model_name=upscale_model_name,
        )

    def _on_cancel(self):
        """Handle Cancel button click."""
        generation_service.cancel()

    def _on_inpaint_mode_changed(self, enabled: bool):
        """Handle Inpaint Mode toggle."""
        self._image_display.set_inpaint_mode(enabled)
        if enabled:
            self._status_bar.set_text("Inpaint mode enabled - select a mask tool to draw")
        else:
            self._status_bar.set_text("Inpaint mode disabled")

    def _on_inpaint_tool_changed(self, tool: InpaintTool):
        """Handle inpaint tool change."""
        # Map toolbar InpaintTool to image display MaskTool
        tool_map = {
            InpaintTool.NONE: MaskTool.NONE,
            InpaintTool.RECT: MaskTool.RECT,
            InpaintTool.PAINT: MaskTool.PAINT,
        }
        mask_tool = tool_map.get(tool, MaskTool.NONE)
        self._image_display.set_mask_tool(mask_tool)

        if tool == InpaintTool.RECT:
            self._status_bar.set_text("Rect Mask: Click and drag to draw rectangular mask")
        elif tool == InpaintTool.PAINT:
            self._status_bar.set_text("Paint Mask: Click and drag to paint mask (25px brush)")
        else:
            self._status_bar.set_text("Select a mask tool to draw")

    def _on_clear_masks(self):
        """Handle Clear Masks button click."""
        self._image_display.clear_masks()
        self._status_bar.set_text("Masks cleared")

    def _on_generate_inpaint(self):
        """Handle Generate Inpaint button click."""
        if not generation_service.is_model_loaded:
            self._status_bar.set_text("Please load models first")
            return

        # Check if there's a mask
        if not self._image_display.has_mask():
            self._status_bar.set_text("Please draw a mask first")
            return

        # Get the original image (stored when entering inpaint mode)
        original_image = self._image_display.get_original_image()
        if original_image is None:
            original_image = self._image_display.get_pil_image()

        if original_image is None:
            self._status_bar.set_text("No image to inpaint")
            return

        # Get the mask
        mask_image = self._image_display.get_mask_image()
        if mask_image is None:
            self._status_bar.set_text("Failed to get mask")
            return

        positive = self._prompt_panel.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        negative = self._prompt_panel.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)
        strength = self._params_widget.get_strength()

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

        # Get upscale settings
        upscale_enabled = self._upscale_widget.is_enabled
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        generation_service.generate_inpaint(
            params,
            input_image=original_image,
            mask_image=mask_image,
            strength=strength,
            upscale_enabled=upscale_enabled,
            upscale_model_path=upscale_model_path,
            upscale_model_name=upscale_model_name,
        )

    def _on_state_changed(self, state: GenerationState):
        """Handle generation state change."""
        self._toolbar.set_state(state)

        if state == GenerationState.IDLE:
            self._toolbar.set_model_loaded(generation_service.is_model_loaded)

    def _on_progress(self, message: str, progress: float):
        """Handle progress update."""
        self._toolbar.set_progress(message, progress)
        self._status_bar.set_text(message)

    def _on_step_progress(self, step: int, total: int):
        """Handle step progress update."""
        self._toolbar.set_step_progress(step, total)

    def _on_generation_complete(self, result: GenerationResult):
        """Handle generation completion."""
        self._toolbar.clear_progress()

        if result.success and result.image:
            # In inpaint mode, we want to keep the mask and show the result
            # The user can continue to refine with the same mask
            in_inpaint_mode = self._image_display.inpaint_mode

            # Display the generated image
            self._image_display.set_image(result.image)

            # Add to thumbnail gallery
            if result.path:
                self._thumbnail_gallery.add_image(result.path, result.image)

            # Only update seed widget if user specified a seed (not random)
            # This preserves -1 for random seed behavior
            if not self._last_seed_was_random and result.seed != -1:
                self._params_widget.set_seed(result.seed)

            # Show the actual seed used in status bar
            seed_info = f" (seed: {result.seed})" if result.seed != -1 else ""
            mode_info = " (inpaint)" if in_inpaint_mode else ""
            self._status_bar.set_text(f"Generated: {result.path.name if result.path else 'image'}{seed_info}{mode_info}")

            # Update toolbar has_image state
            self._toolbar.set_has_image(True)
        else:
            self._status_bar.set_text(f"Generation failed: {result.error or 'Unknown error'}")

    def _on_thumbnail_selected(self, path: Path):
        """Handle thumbnail selection."""
        # Exit inpaint mode when selecting a different image
        if self._toolbar.inpaint_mode:
            self._toolbar.exit_inpaint_mode()

        self._image_display.set_image_from_path(path)

        # Update toolbar has_image state
        self._toolbar.set_has_image(True)

        # Try to load metadata and restore parameters
        metadata = load_metadata_from_image(path)

        if metadata:
            # Restore prompts
            self._prompt_panel.set_prompts(
                metadata.prompt,
                metadata.negative_prompt
            )

            # Restore generation parameters
            self._params_widget.set_size(metadata.width, metadata.height)
            self._params_widget.set_steps(metadata.steps)
            self._params_widget.set_cfg_scale(metadata.cfg_scale)
            self._params_widget.set_seed(metadata.seed)
            self._params_widget.set_sampler(metadata.sampler)

            # Try to select models by name (if available in current model list)
            if metadata.checkpoint:
                self._checkpoint_selector.set_selected_by_name(metadata.checkpoint)
            if metadata.vae:
                self._vae_selector.set_selected_by_name(metadata.vae)
            if metadata.clip:
                self._clip_selector.set_selected_by_name(metadata.clip)

            # Restore upscale settings
            self._upscale_widget.set_enabled(metadata.upscale_enabled)
            if metadata.upscale_model:
                self._upscale_widget.set_model_by_name(metadata.upscale_model)

            self._status_bar.set_text(f"Loaded: {path.name} (parameters restored)")
        else:
            # No metadata - reset to defaults
            self._params_widget.reset_to_defaults()
            self._upscale_widget.reset_to_defaults()
            self._status_bar.set_text(f"Loaded: {path.name} (no metadata - using defaults)")
