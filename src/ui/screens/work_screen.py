"""Main work screen with 3-panel layout."""

from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from src.core.config import config_manager
from src.core.model_manager import model_manager, ModelType
from src.core.generation_service import generation_service, GenerationState, GenerationResult
from src.backends.upscale_backend import upscale_backend
from src.backends.diffusers_backend import diffusers_backend, DiffusersBackend, GenerationParams
from src.ui.widgets.model_selector import ModelSelector
from src.ui.widgets.vram_display import VRAMDisplay
from src.ui.widgets.generation_params import GenerationParamsWidget
from src.ui.widgets.prompt_section import PromptSection
from src.ui.widgets.image_display import ImageDisplayFrame
from src.ui.widgets.thumbnail_gallery import ThumbnailGallery
from src.ui.widgets.toolbar import Toolbar
from src.ui.widgets.upscale_settings import UpscaleSettingsWidget
from src.ui.widgets.batch_settings import BatchSettingsWidget
from src.ui.widgets.generation_progress import GenerationProgressWidget
from src.ui.widgets.toolbar import InpaintTool, OutpaintTool, CropTool, RefinerTool
from src.ui.widgets.image_display import MaskTool, OutpaintTool as ImageOutpaintTool
from src.backends.segmentation_backend import segmentation_backend, DetectedMask
from src.ui.widgets.lora_selector import LoRASelectorPanel
from src.ui.widgets.info_helper import SectionHeader, add_hover_tooltip, SECTION_INFO, LABEL_TOOLTIPS
from src.ui.widgets.collapsible_panel import PanedCollapseButton
from src.utils.metadata import load_metadata_from_image


class WorkScreen(Gtk.Box):
    """Main work screen with 3-panel layout for image generation."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._last_seed_was_random = False
        # Track pending generation to run after model loading
        # Values: None, "generate", "img2img", "inpaint"
        self._pending_generation = None
        # Batch generation state
        self._batch_mode = False
        self._batch_type = None  # "generate" or "img2img"
        self._batch_count = 0
        self._batch_current = 0
        self._batch_params = None
        self._batch_input_image = None  # For img2img batch
        # Multi-GPU batch generation
        self._gpu_backends: dict[int, DiffusersBackend] = {}  # GPU index -> backend instance
        self._batch_executor: ThreadPoolExecutor = None
        self._batch_cancelled = False
        self._batch_completed = 0  # Track actual completed count
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
            on_upscale=self._on_upscale,
            on_cancel=self._on_cancel,
            on_inpaint_mode_changed=self._on_inpaint_mode_changed,
            on_inpaint_tool_changed=self._on_inpaint_tool_changed,
            on_clear_masks=self._on_clear_masks,
            on_generate_inpaint=self._on_generate_inpaint,
            on_outpaint_mode_changed=self._on_outpaint_mode_changed,
            on_outpaint_tool_changed=self._on_outpaint_tool_changed,
            on_clear_outpaint_masks=self._on_clear_outpaint_masks,
            on_generate_outpaint=self._on_generate_outpaint,
            on_crop_mode_changed=self._on_crop_mode_changed,
            on_crop_tool_changed=self._on_crop_tool_changed,
            on_clear_crop_mask=self._on_clear_crop_mask,
            on_crop_image=self._on_crop_image,
            on_crop_size_changed=self._on_crop_size_changed,
            on_remove_with_mask=self._on_remove_with_mask,
            on_refiner_mode_changed=self._on_refiner_mode_changed,
            on_refiner_detect=self._on_refiner_detect,
            on_clear_refiner_masks=self._on_clear_refiner_masks,
            on_generate_refine=self._on_generate_refine,
        )
        self.append(self._toolbar)

        # Main content area with 3 panels
        self._paned_outer = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self._paned_outer.set_vexpand(True)
        self.append(self._paned_outer)

        # Get window config for collapsed states
        window_config = config_manager.config.window

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

        # Restore panel sizes from config
        self._paned_outer.set_position(window_config.left_panel_width)
        self._paned_inner.set_position(window_config.right_panel_position)

        # Create and add collapse buttons to their containers
        # Left panel collapse button (on the right edge of left panel)
        self._left_collapse_btn = PanedCollapseButton(
            self._paned_outer,
            collapse_direction="left",
            initially_collapsed=window_config.left_panel_collapsed,
        )
        self._left_collapse_btn.set_saved_position(window_config.left_panel_width)
        self._left_collapse_btn.set_valign(Gtk.Align.CENTER)
        self._left_panel_container.append(self._left_collapse_btn)

        # Right panel collapse button (on the left edge of right panel)
        self._right_collapse_btn = PanedCollapseButton(
            self._paned_inner,
            collapse_direction="right",
            initially_collapsed=window_config.right_panel_collapsed,
        )
        self._right_collapse_btn.set_saved_position(window_config.right_panel_position)
        self._right_collapse_btn.set_valign(Gtk.Align.CENTER)
        self._right_panel_container.prepend(self._right_collapse_btn)

        # Prompt panel collapse button (at the top of prompt section)
        self._prompt_collapse_btn = PanedCollapseButton(
            self._center_paned,
            collapse_direction="bottom",
            initially_collapsed=window_config.prompt_panel_collapsed,
        )
        self._prompt_collapse_btn.set_saved_position(window_config.center_panel_height)
        self._prompt_collapse_btn.set_halign(Gtk.Align.CENTER)
        self._prompt_panel_container.prepend(self._prompt_collapse_btn)

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
        # Container with scrolled content and collapse button
        container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.add_css_class("left-panel")
        scrolled.set_size_request(200, -1)  # Minimum width
        scrolled.set_hexpand(True)
        container.append(scrolled)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(8)
        box.set_margin_end(8)
        scrolled.set_child(box)

        # Store reference to add collapse button later
        self._left_panel_container = container

        # VRAM display (at the top)
        self._vram_display = VRAMDisplay()
        box.append(self._vram_display)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(6)
        separator.set_margin_bottom(6)
        box.append(separator)

        # Model selectors section with info button
        model_header = SectionHeader("Models", SECTION_INFO["models"])
        box.append(model_header)

        # Checkpoint row: selector + optimize checkbox
        checkpoint_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        box.append(checkpoint_row)

        self._checkpoint_selector = ModelSelector(
            label="Checkpoint",
            model_type=ModelType.CHECKPOINT,
            compact=True,
            label_width=65,
        )
        self._checkpoint_selector.set_hexpand(True)
        checkpoint_row.append(self._checkpoint_selector)

        # Optimize checkbox for torch.compile
        self._optimize_checkbox = Gtk.CheckButton(label="Optimize")
        self._optimize_checkbox.add_css_class("caption")
        self._optimize_checkbox.set_active(False)  # Default to disabled
        add_hover_tooltip(self._optimize_checkbox, LABEL_TOOLTIPS["optimize"])
        checkpoint_row.append(self._optimize_checkbox)

        # VAE selector
        self._vae_selector = ModelSelector(
            label="VAE",
            model_type=ModelType.VAE,
            compact=True,
            label_width=65,
        )
        box.append(self._vae_selector)

        # CLIP selector
        self._clip_selector = ModelSelector(
            label="CLIP",
            model_type=ModelType.CLIP,
            compact=True,
            label_width=65,
        )
        box.append(self._clip_selector)

        # Separator
        separator2 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator2.set_margin_top(6)
        separator2.set_margin_bottom(6)
        box.append(separator2)

        # Generation parameters
        self._params_widget = GenerationParamsWidget()
        box.append(self._params_widget)

        # Separator
        separator3 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator3.set_margin_top(6)
        separator3.set_margin_bottom(6)
        box.append(separator3)

        # LoRA selector panel
        self._lora_panel = LoRASelectorPanel(
            on_changed=self._on_lora_changed,
        )
        box.append(self._lora_panel)

        # Separator
        separator4 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator4.set_margin_top(6)
        separator4.set_margin_bottom(6)
        box.append(separator4)

        # Upscale settings
        self._upscale_widget = UpscaleSettingsWidget(
            on_changed=self._on_upscale_settings_changed
        )
        box.append(self._upscale_widget)

        # Separator
        separator5 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator5.set_margin_top(6)
        separator5.set_margin_bottom(6)
        box.append(separator5)

        # Batch settings
        self._batch_widget = BatchSettingsWidget(
            on_changed=self._on_batch_settings_changed
        )
        box.append(self._batch_widget)

        # Separator
        separator6 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator6.set_margin_top(6)
        separator6.set_margin_bottom(6)
        box.append(separator6)

        # Generation progress
        self._progress_widget = GenerationProgressWidget()
        box.append(self._progress_widget)

        return container

    def _create_center_panel(self) -> Gtk.Widget:
        """Create the center panel with image display and prompts."""
        # Main container for center panel
        center_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        center_box.add_css_class("center-panel")
        center_box.set_margin_top(12)
        center_box.set_margin_bottom(12)
        center_box.set_margin_start(12)
        center_box.set_margin_end(12)

        # Use vertical paned to allow resizing between image and prompts
        self._center_paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self._center_paned.set_vexpand(True)
        center_box.append(self._center_paned)

        # Image display (top)
        self._image_display = ImageDisplayFrame()
        self._image_display.set_vexpand(True)
        self._image_display.set_size_request(-1, 200)  # Minimum height
        self._image_display.set_on_image_dropped(self._on_image_dropped)
        self._center_paned.set_start_child(self._image_display)
        self._center_paned.set_resize_start_child(True)
        self._center_paned.set_shrink_start_child(True)

        # Prompt section (bottom) - contains prompt manager + prompts
        # Container with collapse button at top
        prompt_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._prompt_panel_container = prompt_container

        self._prompt_section = PromptSection()
        self._prompt_section.set_size_request(-1, 150)  # Minimum height
        self._prompt_section.set_vexpand(True)
        prompt_container.append(self._prompt_section)

        self._center_paned.set_end_child(prompt_container)
        self._center_paned.set_resize_end_child(True)
        self._center_paned.set_shrink_end_child(True)

        # Restore center panel position from config
        window_config = config_manager.config.window
        self._center_paned.set_position(window_config.center_panel_height)

        # Restore prompt section positions from config
        if window_config.prompt_section_width > 0:
            self._prompt_section.set_paned_positions({
                "prompts_width": window_config.prompt_section_width,
            })
        if window_config.prompt_section_split > 0:
            self._prompt_section.set_paned_positions({
                "prompts_split": window_config.prompt_section_split,
            })
        if window_config.prompt_manager_split > 0:
            self._prompt_section.prompt_manager.set_paned_position(window_config.prompt_manager_split)

        # Restore prompt font sizes from config
        self._prompt_section.set_font_sizes(
            window_config.positive_prompt_font_size,
            window_config.negative_prompt_font_size,
            window_config.refiner_prompt_font_size,
        )

        return center_box

    def _create_right_panel(self) -> Gtk.Widget:
        """Create the right panel with thumbnail gallery."""
        # Container with collapse button and scrolled content
        container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)

        # Store reference to add collapse button later
        self._right_panel_container = container

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.add_css_class("right-panel")
        scrolled.set_size_request(150, -1)  # Minimum width
        scrolled.set_hexpand(True)
        container.append(scrolled)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(8)
        box.set_margin_end(8)
        scrolled.set_child(box)

        # Thumbnail gallery
        self._thumbnail_gallery = ThumbnailGallery(
            on_image_selected=self._on_thumbnail_selected,
            on_directory_changed=self._on_gallery_directory_changed,
            on_image_deleted=self._on_image_deleted,
        )
        self._thumbnail_gallery.set_vexpand(True)
        box.append(self._thumbnail_gallery)

        return container

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
        self._update_upscale_button_state()

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
        # Also scan for LoRAs
        self._lora_panel.scan_loras()
        self._status_bar.set_text("Ready")

    def _on_load_models(self):
        """Handle Load Models button click."""
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
            return

        # Get current params for resolution
        params = self._params_widget.get_params("", "")
        width, height = params.width, params.height

        # Check if optimize is enabled and we have a compiled cache
        use_compiled = self._optimize_checkbox.get_active()
        checkpoint_path = str(model_manager.loaded.checkpoint.path)

        if use_compiled:
            has_compiled = diffusers_backend.has_compiled_cache(checkpoint_path, width, height)
            if has_compiled:
                self._status_bar.set_text(f"Loading optimized model for {width}x{height}...")
            else:
                self._status_bar.set_text("Loading model...")
                use_compiled = False  # No cache, load without compile for now
        else:
            self._status_bar.set_text("Loading model...")

        generation_service.load_models(
            use_compiled=use_compiled,
            target_width=width,
            target_height=height,
        )

    def _on_clear_models(self):
        """Handle Clear Models button click."""
        self._pending_generation = None  # Clear any pending generation
        generation_service.unload_models()
        self._toolbar.set_model_loaded(False)
        self._status_bar.set_text("Models unloaded")

    def _on_gallery_directory_changed(self, directory: Path):
        """Handle gallery directory change."""
        if directory.exists():
            self._status_bar.set_text(f"Output directory: {directory.name or 'root'}")
        else:
            self._status_bar.set_text(f"New directory: {directory.name} (will be created on save)")

    def _on_image_deleted(self, path: Path):
        """Handle image deletion from gallery."""
        # Exit inpaint mode if active
        if self._toolbar.inpaint_mode:
            self._toolbar.exit_inpaint_mode()

        # Check if gallery selected a new image after deletion
        new_selected = self._thumbnail_gallery.get_selected_path()
        if new_selected and new_selected.exists():
            # Load the newly selected image
            self._on_thumbnail_selected(new_selected)
            self._status_bar.set_text(f"Deleted: {path.name}")
        else:
            # No more images - clear the display
            self._image_display.clear()

            # Update toolbar state
            self._toolbar.set_has_image(False)
            self._update_upscale_button_state()

            # Clear prompts to defaults
            self._prompt_section.set_prompts("", "")
            self._params_widget.reset_to_defaults()

            self._status_bar.set_text(f"Deleted: {path.name}")

    def _on_lora_changed(self):
        """Handle LoRA selection change."""
        active_loras = self._lora_panel.get_active_loras()
        if active_loras:
            names = [lora.name for lora in active_loras]
            self._status_bar.set_text(f"LoRAs: {', '.join(names)}")
        else:
            self._status_bar.set_text("No LoRAs selected")

    def _get_active_loras(self) -> list:
        """Get list of active LoRAs as (path, name, weight) tuples."""
        active_loras = self._lora_panel.get_active_loras()
        return [(lora.path, lora.name, lora.weight) for lora in active_loras]

    def _needs_model_reload(self) -> bool:
        """Check if models need to be (re)loaded based on selection changes."""
        # If no model loaded at all, need to load
        if not generation_service.is_model_loaded:
            return True

        # Check if selected checkpoint differs from loaded checkpoint
        selected_checkpoint = model_manager.loaded.checkpoint
        if selected_checkpoint:
            selected_path = str(selected_checkpoint.path)
            loaded_path = diffusers_backend.loaded_checkpoint
            if selected_path != loaded_path:
                return True

        return False

    def _needs_optimization(self) -> bool:
        """Check if optimization (torch.compile) is needed."""
        if not self._optimize_checkbox.get_active():
            return False

        if not model_manager.loaded.checkpoint:
            return False

        # Check if model is already compiled
        if diffusers_backend.is_compiled:
            return False

        return True

    def _ensure_model_ready(self, pending_action: str) -> bool:
        """
        Ensure model is loaded and optimized if needed.

        Args:
            pending_action: The action to perform after model is ready
                           ("generate", "img2img", "inpaint")

        Returns:
            True if generation can proceed immediately, False if waiting for load/compile
        """
        params = self._params_widget.get_params("", "")
        width, height = params.width, params.height
        checkpoint_path = str(model_manager.loaded.checkpoint.path)
        optimize_enabled = self._optimize_checkbox.get_active()

        # Check if we need to load or reload the model
        needs_reload = self._needs_model_reload()

        if needs_reload:
            if optimize_enabled:
                has_compiled = diffusers_backend.has_compiled_cache(checkpoint_path, width, height)
                if has_compiled:
                    # Load with existing compiled cache
                    self._pending_generation = pending_action
                    self._status_bar.set_text(f"Loading optimized model for {width}x{height}...")
                    generation_service.load_models(
                        use_compiled=True,
                        target_width=width,
                        target_height=height,
                    )
                    return False
                else:
                    # Need to compile first
                    self._pending_generation = pending_action
                    self._run_optimization_then_generate(width, height)
                    return False
            else:
                # Load without optimization
                self._pending_generation = pending_action
                self._on_load_models()
                return False

        # Model is loaded, check if we need optimization
        if optimize_enabled and not diffusers_backend.is_compiled:
            has_compiled = diffusers_backend.has_compiled_cache(checkpoint_path, width, height)
            if has_compiled:
                # Reload with compiled cache
                self._pending_generation = pending_action
                self._status_bar.set_text(f"Loading optimized model for {width}x{height}...")
                generation_service.load_models(
                    use_compiled=True,
                    target_width=width,
                    target_height=height,
                )
                return False
            else:
                # Need to compile
                self._pending_generation = pending_action
                self._run_optimization_then_generate(width, height)
                return False

        # Model is ready
        return True

    def _run_optimization_then_generate(self, width: int, height: int):
        """Run torch.compile optimization, then proceed with pending generation."""
        self._status_bar.set_text(f"Optimizing model for {width}x{height} (first time only)...")
        self._toolbar.set_state(GenerationState.LOADING)

        def compile_thread():
            try:
                success = diffusers_backend.compile_model(
                    checkpoint_path=str(model_manager.loaded.checkpoint.path),
                    model_type=model_manager.loaded.checkpoint.components.model_type,
                    vae_path=str(model_manager.loaded.vae.path) if model_manager.loaded.vae else None,
                    target_width=width,
                    target_height=height,
                    progress_callback=lambda msg, prog: GLib.idle_add(
                        self._on_optimization_progress, msg, prog
                    ),
                )
                GLib.idle_add(self._on_optimization_complete, success)
            except Exception as e:
                print(f"Error during optimization: {e}")
                import traceback
                traceback.print_exc()
                GLib.idle_add(self._on_optimization_complete, False, str(e))

        thread = threading.Thread(target=compile_thread, daemon=True)
        thread.start()

    def _on_optimization_progress(self, message: str, progress: float):
        """Handle optimization progress update."""
        self._status_bar.set_text(message)
        self._progress_widget.set_status(message)
        self._progress_widget.set_step_fraction(progress)

    def _on_optimization_complete(self, success: bool, error: str = None):
        """Handle optimization completion."""
        self._progress_widget.reset()
        self._toolbar.set_state(GenerationState.IDLE)

        if success:
            self._status_bar.set_text("Optimization complete, starting generation...")
            self._toolbar.set_model_loaded(True)

            # Execute the pending generation
            pending = self._pending_generation
            self._pending_generation = None
            if pending == "generate":
                self._do_generate()
            elif pending == "img2img":
                self._do_img2img()
            elif pending == "inpaint":
                self._do_generate_inpaint()
            elif pending == "refine":
                selected_masks = self._image_display.get_selected_refiner_masks()
                if selected_masks:
                    self._do_generate_refine(selected_masks)
        else:
            self._pending_generation = None
            self._status_bar.set_text(f"Optimization failed: {error or 'Unknown error'}")

    def _on_generate(self):
        """Handle Generate button click."""
        # Check if a checkpoint is selected
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
            return

        positive = self._prompt_section.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        # For batch mode, skip normal model loading - batch handles its own
        batch_count = self._batch_widget.batch_count
        if batch_count > 1:
            self._start_batch_generation("generate")
            return

        # Clean up batch backends before single generation to free VRAM
        if self._gpu_backends:
            self._status_bar.set_text("Cleaning up batch backends...")
            self._cleanup_gpu_backends()

        # Ensure model is loaded and optimized if needed (single image only)
        if not self._ensure_model_ready("generate"):
            return

        self._do_generate()

    def _do_generate(self):
        """Perform the actual generation (called after models are loaded, single image only)."""
        positive = self._prompt_section.get_positive_prompt()
        negative = self._prompt_section.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

        # Set up progress widget for single image (0/1 at start, 1/1 when complete)
        self._progress_widget.set_batch_progress(0, 1)
        self._progress_widget.set_status("Starting generation...")
        self._progress_widget.set_generating(True)

        # Get upscale settings
        upscale_enabled = self._upscale_widget.is_enabled
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        # Get output directory from gallery
        output_dir = self._thumbnail_gallery.get_output_directory()

        # Get active LoRAs
        loras = self._get_active_loras()

        generation_service.generate(
            params,
            upscale_enabled=upscale_enabled,
            upscale_model_path=upscale_model_path,
            upscale_model_name=upscale_model_name,
            output_dir=output_dir,
            loras=loras if loras else None,
        )

    def _start_batch_generation(self, batch_type: str):
        """Start a parallel batch generation sequence using multiple GPUs."""
        self._batch_mode = True
        self._batch_type = batch_type
        self._batch_count = self._batch_widget.batch_count
        self._batch_current = 0
        self._batch_cancelled = False
        self._batch_completed = 0

        # Store input image for img2img batch (use the same image for all)
        if batch_type == "img2img":
            self._batch_input_image = self._image_display.get_pil_image()

        # Get selected GPUs
        gpu_indices = self._batch_widget.selected_gpu_indices
        if not gpu_indices:
            gpu_indices = [0]  # Default to GPU 0

        self._status_bar.set_text(f"Starting batch generation: {self._batch_count} images on {len(gpu_indices)} GPU(s)...")

        # Set up progress widget with per-GPU step progress bars
        self._progress_widget.set_batch_progress(0, self._batch_count)
        self._progress_widget.setup_gpu_progress_bars(gpu_indices)
        self._progress_widget.set_status("Loading models...")
        self._progress_widget.set_generating(True)

        # Disable batch widget and toolbar during generation
        self._batch_widget.set_sensitive_all(False)
        self._toolbar.set_state(GenerationState.GENERATING)

        # Read optimization setting on main thread (GTK widgets not thread-safe)
        optimize_enabled = self._optimize_checkbox.get_active()

        # Start batch generation in a background thread
        thread = threading.Thread(
            target=self._run_parallel_batch,
            args=(batch_type, gpu_indices, optimize_enabled),
            daemon=True
        )
        thread.start()

    def _run_parallel_batch(self, batch_type: str, gpu_indices: list[int], optimize_enabled: bool):
        """Run parallel batch generation across multiple GPUs."""
        import gc
        import torch
        from PIL import Image

        # Initialize completed counter before try block so it's accessible in finally
        completed = 0

        try:
            # Get generation parameters
            # Store raw user prompt and negative prompt (without random words)
            raw_positive = self._prompt_section.get_raw_positive_prompt()
            negative = self._prompt_section.get_negative_prompt()
            # Get a sample params for dimensions/settings (prompt will be replaced per-generation)
            params = self._params_widget.get_params(raw_positive, negative)
            strength = self._params_widget.get_strength()
            # Store reference to prompt manager for generating fresh random prompts
            prompt_manager = self._prompt_section.prompt_manager

            # Get model info
            checkpoint_path = str(model_manager.loaded.checkpoint.path)
            model_type = model_manager.loaded.checkpoint.components.model_type
            vae_path = str(model_manager.loaded.vae.path) if model_manager.loaded.vae else None

            # Get LoRAs
            loras = self._get_active_loras()

            # Get output directory
            output_dir = self._thumbnail_gallery.get_output_directory()

            # Get upscale settings
            upscale_enabled = self._upscale_widget.is_enabled
            upscale_model_path = self._upscale_widget.selected_model_path
            upscale_model_name = self._upscale_widget.selected_model_name if upscale_enabled else ""

            # Get model names for metadata
            checkpoint_name = model_manager.loaded.checkpoint.name
            vae_name = model_manager.loaded.vae.name if model_manager.loaded.vae else ""

            # Check optimization settings
            # optimize_enabled is passed from main thread (GTK widgets not thread-safe)
            # NOTE: For multi-GPU batch, only the first GPU uses torch.compile to avoid
            # FX tracing conflicts when running compiled models concurrently
            use_compiled = False
            single_gpu_batch = len(gpu_indices) == 1

            if optimize_enabled:
                # Check if compiled cache exists
                has_cache = diffusers_backend.has_compiled_cache(
                    checkpoint_path, params.width, params.height
                )
                print(f"[Batch] optimize_enabled={optimize_enabled}, has_cache={has_cache}, resolution={params.width}x{params.height}")

                if has_cache:
                    gpu_str = f"GPU {gpu_indices[0]}" if len(gpu_indices) == 1 else f"{len(gpu_indices)} GPUs"
                    GLib.idle_add(
                        self._status_bar.set_text,
                        f"Compiled cache found. Loading optimized model on {gpu_str}..."
                    )
                    use_compiled = True
                else:
                    # Need to compile first - only compile once on first GPU
                    # The compiled kernels are cached on disk and shared by all GPUs
                    GLib.idle_add(
                        self._status_bar.set_text,
                        f"Compiling model on GPU {gpu_indices[0]} (first time only, cache shared by all GPUs)..."
                    )

                    # Create a temporary backend on first GPU for compilation
                    compile_backend = DiffusersBackend(gpu_index=gpu_indices[0])
                    success = compile_backend.compile_model(
                        checkpoint_path=checkpoint_path,
                        model_type=model_type,
                        vae_path=vae_path,
                        target_width=params.width,
                        target_height=params.height,
                        progress_callback=lambda msg, prog: GLib.idle_add(
                            self._on_batch_compile_progress, msg, prog
                        ),
                    )

                    # Unload the compile backend to free VRAM
                    compile_backend.unload_model()
                    del compile_backend

                    # Force cleanup before loading on multiple GPUs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if not success:
                        GLib.idle_add(
                            self._status_bar.set_text,
                            "Compilation failed. Proceeding without optimization..."
                        )
                        use_compiled = False
                    else:
                        gpu_str = f"GPU {gpu_indices[0]}" if len(gpu_indices) == 1 else f"{len(gpu_indices)} GPUs"
                        GLib.idle_add(
                            self._status_bar.set_text,
                            f"Compilation complete. Loading optimized model on {gpu_str}..."
                        )
                        use_compiled = True

            if optimize_enabled and not single_gpu_batch:
                # Multi-GPU with optimization: only first GPU gets compiled to avoid FX tracing conflicts
                GLib.idle_add(
                    self._status_bar.set_text,
                    f"Loading models on {len(gpu_indices)} GPU(s) (optimized on GPU {gpu_indices[0]})..."
                )
            elif not optimize_enabled and len(gpu_indices) == 1:
                GLib.idle_add(self._status_bar.set_text, f"Loading model on GPU {gpu_indices[0]}...")
            elif not optimize_enabled and len(gpu_indices) > 1:
                GLib.idle_add(self._status_bar.set_text, f"Loading models on {len(gpu_indices)} GPU(s)...")

            # Reuse existing backend instances if possible, or create new ones
            # For multi-GPU with optimization: only compile on first GPU to avoid FX tracing conflicts
            backends_to_remove = set(self._gpu_backends.keys()) - set(gpu_indices)
            for gpu_idx in backends_to_remove:
                # Remove backends for GPUs no longer in use
                try:
                    self._gpu_backends[gpu_idx].unload_model()
                except Exception:
                    pass
                del self._gpu_backends[gpu_idx]

            for i, gpu_idx in enumerate(gpu_indices):
                if self._batch_cancelled:
                    break

                # Only use torch.compile on the first GPU when multi-GPU optimization is enabled
                # This avoids FX tracing conflicts while still benefiting from compilation on one GPU
                compile_this_gpu = use_compiled and (single_gpu_batch or i == 0)
                print(f"[Batch] GPU {gpu_idx} (i={i}): use_compiled={use_compiled}, single_gpu_batch={single_gpu_batch}, compile_this_gpu={compile_this_gpu}")

                # Check if we can reuse an existing backend
                existing_backend = self._gpu_backends.get(gpu_idx)
                if existing_backend and existing_backend.is_loaded:
                    # Check if the loaded model matches what we need
                    same_checkpoint = existing_backend.loaded_checkpoint == checkpoint_path
                    same_vae = existing_backend.loaded_vae == vae_path
                    same_compiled = existing_backend.is_compiled == compile_this_gpu
                    print(f"[Batch] GPU {gpu_idx}: Checking reuse - same_ckpt={same_checkpoint}, same_vae={same_vae}, same_compiled={same_compiled} (existing={existing_backend.is_compiled}, needed={compile_this_gpu})")

                    if same_checkpoint and same_vae and same_compiled:
                        # Can reuse this backend - just update LoRAs if needed
                        print(f"[Batch] GPU {gpu_idx}: Reusing existing backend")
                        GLib.idle_add(
                            self._status_bar.set_text,
                            f"Reusing {'optimized ' if compile_this_gpu else ''}model on GPU {gpu_idx}..."
                        )
                        # TODO: Handle LoRA changes if needed
                        continue
                else:
                    print(f"[Batch] GPU {gpu_idx}: No existing backend to reuse")

                # Need to load fresh
                GLib.idle_add(
                    self._status_bar.set_text,
                    f"Loading {'optimized ' if compile_this_gpu else ''}model on GPU {gpu_idx} ({i + 1}/{len(gpu_indices)})..."
                )

                # Unload existing backend if any
                if existing_backend:
                    try:
                        existing_backend.unload_model()
                    except Exception:
                        pass

                backend = DiffusersBackend(gpu_index=gpu_idx)
                success = backend.load_model(
                    checkpoint_path=checkpoint_path,
                    model_type=model_type,
                    vae_path=vae_path,
                    use_compiled=compile_this_gpu,
                    target_width=params.width,
                    target_height=params.height,
                )

                if not success:
                    GLib.idle_add(
                        self._status_bar.set_text,
                        f"Failed to load model on GPU {gpu_idx}"
                    )
                    continue

                # Load LoRAs if any
                if loras:
                    backend.load_loras(loras)

                self._gpu_backends[gpu_idx] = backend

            if not self._gpu_backends:
                GLib.idle_add(self._end_batch_generation)
                return

            GLib.idle_add(
                self._status_bar.set_text,
                f"Models loaded on {len(self._gpu_backends)} GPU(s). Starting generation..."
            )

            # Create a queue of generation tasks
            remaining = self._batch_count
            gpu_list = list(self._gpu_backends.keys())

            # Use persistent ThreadPoolExecutor to run generations in parallel
            # This preserves CUDA warmup state across batch runs
            num_workers = len(self._gpu_backends)
            if self._batch_executor is None or self._batch_executor._max_workers < num_workers:
                # Create or recreate pool if we need more workers
                if self._batch_executor is not None:
                    self._batch_executor.shutdown(wait=False)
                self._batch_executor = ThreadPoolExecutor(max_workers=max(num_workers, 4))

            executor = self._batch_executor
            futures = {}

            while remaining > 0 or futures:
                if self._batch_cancelled:
                    # Cancel pending futures (only affects not-yet-started futures)
                    for future in futures:
                        future.cancel()

                    # Wait for any running futures to complete before cleanup
                    # This prevents segfault from unloading models while generation is in progress
                    GLib.idle_add(self._status_bar.set_text, "Waiting for current generation to finish...")
                    for future in list(futures.keys()):
                        gpu_idx = futures[future]
                        if not future.cancelled():
                            try:
                                # Wait for the running future to complete and process result
                                result_path, result_image = future.result(timeout=120)
                                completed += 1

                                # Reset GPU step progress
                                GLib.idle_add(
                                    self._progress_widget.reset_gpu_step_progress, gpu_idx
                                )

                                # Update UI with result (so gallery gets updated)
                                if result_path and result_image:
                                    GLib.idle_add(
                                        self._on_batch_image_complete,
                                        result_path, result_image, completed, self._batch_count
                                    )
                            except Exception as e:
                                print(f"Error waiting for future during cancellation: {e}")

                    GLib.idle_add(self._status_bar.set_text, f"Batch cancelled after {completed} images")
                    break

                # Submit new tasks for available GPUs
                while remaining > 0 and len(futures) < len(self._gpu_backends):
                    # Find an available GPU
                    busy_gpus = {futures[f] for f in futures}
                    available_gpus = [g for g in gpu_list if g not in busy_gpus]

                    if not available_gpus:
                        break

                    gpu_idx = available_gpus[0]
                    backend = self._gpu_backends[gpu_idx]

                    # Generate fresh random prompt for this generation
                    checked_words = prompt_manager.get_checked_words_string()
                    if checked_words and raw_positive:
                        fresh_prompt = f"{checked_words}, {raw_positive}"
                    elif checked_words:
                        fresh_prompt = checked_words
                    else:
                        fresh_prompt = raw_positive

                    # Create params with random seed and fresh prompt for this generation
                    gen_params = GenerationParams(
                        prompt=fresh_prompt,
                        negative_prompt=params.negative_prompt,
                        width=params.width,
                        height=params.height,
                        steps=params.steps,
                        cfg_scale=params.cfg_scale,
                        seed=-1,  # Random seed for each
                        sampler=params.sampler,
                        scheduler=params.scheduler,
                    )

                    # Submit generation task
                    if batch_type == "generate":
                        future = executor.submit(
                            self._generate_on_gpu,
                            backend, gen_params, output_dir, upscale_enabled, upscale_model_path,
                            upscale_model_name, checkpoint_name, vae_name, model_type, gpu_idx
                        )
                    else:  # img2img
                        future = executor.submit(
                            self._generate_img2img_on_gpu,
                            backend, gen_params, self._batch_input_image, strength,
                            output_dir, upscale_enabled, upscale_model_path,
                            upscale_model_name, checkpoint_name, vae_name, model_type, gpu_idx
                        )

                    futures[future] = gpu_idx
                    remaining -= 1

                # Wait for at least one to complete
                if futures:
                    done_futures = []
                    for future in list(futures.keys()):
                        if future.done():
                            done_futures.append(future)

                    if not done_futures:
                        # Wait a bit and check again
                        import time
                        time.sleep(0.1)
                        continue

                    for future in done_futures:
                        gpu_idx = futures.pop(future)
                        try:
                            result_path, result_image = future.result()
                            completed += 1

                            # Reset GPU step progress for this GPU (ready for next image)
                            GLib.idle_add(
                                self._progress_widget.reset_gpu_step_progress, gpu_idx
                            )

                            # Update UI with result
                            if result_path and result_image:
                                GLib.idle_add(
                                    self._on_batch_image_complete,
                                    result_path, result_image, completed, self._batch_count
                                )
                        except Exception as e:
                            print(f"Generation error on GPU {gpu_idx}: {e}")
                            completed += 1
                            # Also reset GPU progress on error
                            GLib.idle_add(
                                self._progress_widget.reset_gpu_step_progress, gpu_idx
                            )

                        GLib.idle_add(
                            self._status_bar.set_text,
                            f"Batch progress: {completed}/{self._batch_count}"
                        )

        except Exception as e:
            import traceback
            print(f"Batch generation error: {e}")
            traceback.print_exc()
            GLib.idle_add(self._status_bar.set_text, f"Batch error: {e}")

        finally:
            # Save completed count for end message
            self._batch_completed = completed
            # Don't cleanup GPU backends - keep them loaded for reuse in next batch
            # self._cleanup_gpu_backends()
            GLib.idle_add(self._end_batch_generation)

    def _generate_on_gpu(self, backend: DiffusersBackend, params: GenerationParams,
                         output_dir: Path, upscale_enabled: bool, upscale_model_path: str,
                         upscale_model_name: str, checkpoint_name: str, vae_name: str,
                         model_type: str, gpu_idx: int = 0):
        """Generate a single image on a specific GPU backend."""
        import gc
        import time
        import torch
        from src.utils.metadata import save_image_with_metadata, GenerationMetadata

        # Create progress callback for this GPU
        def progress_callback(step: int, total: int):
            GLib.idle_add(self._progress_widget.set_gpu_step_progress, gpu_idx, step, total)

        gen_start = time.time()
        image = backend.generate(params, progress_callback=progress_callback)
        gen_end = time.time()
        print(f"[GPU{gpu_idx}] backend.generate() took {gen_end - gen_start:.3f}s")

        if image is None:
            return None, None

        # Get actual seed used
        actual_seed = backend.get_actual_seed(params) if params.seed == -1 else params.seed

        # Track original size for metadata
        original_width = image.width
        original_height = image.height
        upscale_factor = 4

        # Upscale if enabled
        if upscale_enabled and upscale_model_path:
            upscale_start = time.time()
            upscaled = upscale_backend.upscale(image, upscale_model_path)
            if upscaled:
                image = upscaled
            print(f"[GPU{gpu_idx}] Upscale took {time.time() - upscale_start:.3f}s")

        # Ensure output directory exists (important for symlinked directories)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"batch_{timestamp}_seed{actual_seed}.png"
        output_path = output_dir / filename

        # Create metadata
        metadata = GenerationMetadata(
            checkpoint=checkpoint_name,
            vae=vae_name,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            width=image.width,
            height=image.height,
            steps=params.steps,
            cfg_scale=params.cfg_scale,
            seed=actual_seed,
            sampler=params.sampler,
            scheduler=params.scheduler,
            model_type=model_type,
            upscale_enabled=upscale_enabled,
            upscale_model=upscale_model_name if upscale_enabled else "",
            upscale_factor=upscale_factor if upscale_enabled else 0,
            original_width=original_width if upscale_enabled else 0,
            original_height=original_height if upscale_enabled else 0,
        )

        save_start = time.time()
        save_image_with_metadata(image, output_path, metadata)
        print(f"[GPU{gpu_idx}] Save took {time.time() - save_start:.3f}s")

        # Note: Don't call gc.collect() or empty_cache() here - it resets CUDA warmup state
        # Memory is managed by keeping backends loaded and reusing them

        return output_path, image

    def _generate_img2img_on_gpu(self, backend: DiffusersBackend, params: GenerationParams,
                                  input_image, strength: float, output_dir: Path,
                                  upscale_enabled: bool, upscale_model_path: str,
                                  upscale_model_name: str, checkpoint_name: str, vae_name: str,
                                  model_type: str, gpu_idx: int = 0):
        """Generate a single img2img image on a specific GPU backend."""
        import gc
        import torch
        from src.utils.metadata import save_image_with_metadata, GenerationMetadata

        # Create progress callback for this GPU
        def progress_callback(step: int, total: int):
            GLib.idle_add(self._progress_widget.set_gpu_step_progress, gpu_idx, step, total)

        image = backend.generate_img2img(params, input_image, strength, progress_callback=progress_callback)
        if image is None:
            return None, None

        # Get actual seed used
        actual_seed = backend.get_actual_seed(params) if params.seed == -1 else params.seed

        # Track original size for metadata
        original_width = image.width
        original_height = image.height
        upscale_factor = 4

        # Upscale if enabled
        if upscale_enabled and upscale_model_path:
            upscaled = upscale_backend.upscale(image, upscale_model_path)
            if upscaled:
                image = upscaled

        # Ensure output directory exists (important for symlinked directories)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"batch_img2img_{timestamp}_seed{actual_seed}.png"
        output_path = output_dir / filename

        # Create metadata
        metadata = GenerationMetadata(
            checkpoint=checkpoint_name,
            vae=vae_name,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            width=image.width,
            height=image.height,
            steps=params.steps,
            cfg_scale=params.cfg_scale,
            seed=actual_seed,
            sampler=params.sampler,
            scheduler=params.scheduler,
            model_type=model_type,
            upscale_enabled=upscale_enabled,
            upscale_model=upscale_model_name if upscale_enabled else "",
            upscale_factor=upscale_factor if upscale_enabled else 0,
            original_width=original_width if upscale_enabled else 0,
            original_height=original_height if upscale_enabled else 0,
            is_img2img=True,
            img2img_strength=strength,
        )

        save_image_with_metadata(image, output_path, metadata)

        # Note: Don't call gc.collect() or empty_cache() here - it resets CUDA warmup state

        return output_path, image

    def _on_batch_image_complete(self, path: Path, image, completed: int, total: int):
        """Handle completion of a single batch image (called on main thread)."""
        # Display the latest image
        self._image_display.set_image(image)

        # Add to thumbnail gallery
        self._thumbnail_gallery.add_image(path, image)

        # Update status
        self._status_bar.set_text(f"Batch progress: {completed}/{total} - {path.name}")

        # Update progress widget
        self._progress_widget.set_batch_progress(completed, total)
        self._progress_widget.set_status(f"Completed {completed}/{total}")

        # Update toolbar
        self._toolbar.set_has_image(True)
        self._update_upscale_button_state()

    def _on_batch_compile_progress(self, message: str, progress: float):
        """Handle compilation progress during batch setup (called on main thread)."""
        self._status_bar.set_text(message)
        self._progress_widget.set_status(message)
        self._progress_widget.set_step_fraction(progress)

    def _cleanup_gpu_backends(self):
        """Clean up GPU backend instances."""
        for gpu_idx, backend in self._gpu_backends.items():
            try:
                backend.unload_model()
            except Exception as e:
                print(f"Error unloading model from GPU {gpu_idx}: {e}")
        self._gpu_backends.clear()

        # Force garbage collection
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _continue_batch_generation(self):
        """Continue with the next image in the batch (legacy, not used in parallel mode)."""
        pass  # Parallel batch handles continuation internally

    def _end_batch_generation(self):
        """End the batch generation sequence."""
        total_planned = self._batch_count
        actual_completed = self._batch_completed
        was_cancelled = self._batch_cancelled

        self._batch_mode = False
        self._batch_type = None
        self._batch_count = 0
        self._batch_current = 0
        self._batch_input_image = None
        self._batch_cancelled = False
        self._batch_completed = 0

        # Re-enable batch widget and toolbar
        self._batch_widget.set_sensitive_all(True)
        self._toolbar.set_state(GenerationState.IDLE)

        # Reset progress widget
        self._progress_widget.reset()
        self._progress_widget.set_generating(False)

        # Show appropriate status message
        if was_cancelled:
            self._status_bar.set_text(f"Batch cancelled: {actual_completed}/{total_planned} images generated")
        else:
            self._status_bar.set_text(f"Batch complete: {actual_completed} images generated")

    def _on_img2img(self):
        """Handle Image to Image button click."""
        # Check if a checkpoint is selected
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
            return

        # Check if there's an image to use as input
        input_image = self._image_display.get_pil_image()
        if input_image is None:
            self._status_bar.set_text("Please load or generate an image first")
            return

        positive = self._prompt_section.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        # For batch mode, skip normal model loading - batch handles its own
        batch_count = self._batch_widget.batch_count
        if batch_count > 1:
            self._start_batch_generation("img2img")
            return

        # Clean up batch backends before single generation to free VRAM
        if self._gpu_backends:
            self._status_bar.set_text("Cleaning up batch backends...")
            self._cleanup_gpu_backends()

        # Ensure model is loaded and optimized if needed (single image only)
        if not self._ensure_model_ready("img2img"):
            return

        self._do_img2img()

    def _do_img2img(self):
        """Perform the actual img2img generation (called after models are loaded, single image only)."""
        input_image = self._image_display.get_pil_image()

        positive = self._prompt_section.get_positive_prompt()
        negative = self._prompt_section.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)
        strength = self._params_widget.get_strength()

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

        # Set up progress widget for single image (0/1 at start, 1/1 when complete)
        self._progress_widget.set_batch_progress(0, 1)
        self._progress_widget.set_status("Starting img2img...")
        self._progress_widget.set_generating(True)

        # Get upscale settings
        upscale_enabled = self._upscale_widget.is_enabled
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        # Get output directory from gallery
        output_dir = self._thumbnail_gallery.get_output_directory()

        # Get active LoRAs
        loras = self._get_active_loras()

        generation_service.generate_img2img(
            params,
            input_image=input_image,
            strength=strength,
            upscale_enabled=upscale_enabled,
            upscale_model_path=upscale_model_path,
            upscale_model_name=upscale_model_name,
            output_dir=output_dir,
            loras=loras if loras else None,
        )

    def _on_cancel(self):
        """Handle Cancel button click."""
        # Cancel batch mode if active
        if self._batch_mode:
            self._batch_cancelled = True
            self._status_bar.set_text(f"Cancelling batch generation...")
        generation_service.cancel()

    def _on_inpaint_mode_changed(self, enabled: bool):
        """Handle Inpaint Mode toggle."""
        self._image_display.set_inpaint_mode(enabled)
        if enabled:
            self._center_paned.add_css_class("center-panel-edit-mode")
            # Disable batch during inpaint mode
            self._batch_widget.set_sensitive_all(False)
            self._status_bar.set_text("Inpaint mode enabled - select a mask tool to draw")
        else:
            self._center_paned.remove_css_class("center-panel-edit-mode")
            # Re-enable batch when exiting inpaint mode
            self._batch_widget.set_sensitive_all(True)
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
        # Check if a checkpoint is selected
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
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

        positive = self._prompt_section.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        # Clean up batch backends before single generation to free VRAM
        if self._gpu_backends:
            self._status_bar.set_text("Cleaning up batch backends...")
            self._cleanup_gpu_backends()

        # Ensure model is loaded and optimized if needed
        if not self._ensure_model_ready("inpaint"):
            return

        self._do_generate_inpaint()

    def _do_generate_inpaint(self):
        """Perform the actual inpaint generation (called after models are loaded)."""
        original_image = self._image_display.get_original_image()
        if original_image is None:
            original_image = self._image_display.get_pil_image()

        mask_image = self._image_display.get_mask_image()
        positive = self._prompt_section.get_positive_prompt()
        negative = self._prompt_section.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)
        strength = self._params_widget.get_strength()

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

        # Set up progress widget for single image (0/1 at start, 1/1 when complete)
        self._progress_widget.set_batch_progress(0, 1)
        self._progress_widget.set_status("Starting inpaint...")
        self._progress_widget.set_generating(True)

        # Get upscale settings
        upscale_enabled = self._upscale_widget.is_enabled
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        # Get output directory from gallery
        output_dir = self._thumbnail_gallery.get_output_directory()

        # Get active LoRAs
        loras = self._get_active_loras()

        generation_service.generate_inpaint(
            params,
            input_image=original_image,
            mask_image=mask_image,
            strength=strength,
            upscale_enabled=upscale_enabled,
            upscale_model_path=upscale_model_path,
            upscale_model_name=upscale_model_name,
            output_dir=output_dir,
            loras=loras if loras else None,
        )

    def _on_outpaint_mode_changed(self, enabled: bool):
        """Handle Outpaint Mode toggle."""
        self._image_display.set_outpaint_mode(enabled)
        if enabled:
            self._center_paned.add_css_class("center-panel-outpaint-mode")
            # Disable batch during outpaint mode
            self._batch_widget.set_sensitive_all(False)
            # Set edge zone size from config
            from src.core.config import config_manager
            edge_zone = config_manager.config.outpaint.edge_zone_size
            self._image_display.set_edge_zone_size(edge_zone)
            self._status_bar.set_text("Outpaint mode enabled - click Edge Mask to draw extensions")
        else:
            self._center_paned.remove_css_class("center-panel-outpaint-mode")
            # Re-enable batch when exiting outpaint mode
            self._batch_widget.set_sensitive_all(True)
            self._status_bar.set_text("Outpaint mode disabled")

    def _on_outpaint_tool_changed(self, tool):
        """Handle outpaint tool change."""
        if tool == OutpaintTool.EDGE:
            self._image_display.set_outpaint_tool(ImageOutpaintTool.EDGE)
            self._status_bar.set_text("Edge Mask: Click near an edge and drag outward to extend")
        else:
            self._image_display.set_outpaint_tool(OutpaintTool.NONE)
            self._status_bar.set_text("Select Edge Mask tool to draw extensions")

    def _on_clear_outpaint_masks(self):
        """Handle Clear Outpaint Masks button click."""
        self._image_display.clear_outpaint_masks()
        self._status_bar.set_text("Outpaint masks cleared")

    def _on_generate_outpaint(self):
        """Handle Generate Outpaint button click."""
        # Check if a checkpoint is selected
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
            return

        # Check if there are any outpaint extensions
        if not self._image_display.has_outpaint_extensions():
            self._status_bar.set_text("Please draw outpaint extensions first")
            return

        # Check if there's an image to outpaint (use current displayed image)
        if self._image_display.get_pil_image() is None:
            self._status_bar.set_text("No image to outpaint")
            return

        positive = self._prompt_section.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        # Clean up batch backends before single generation to free VRAM
        if self._gpu_backends:
            self._status_bar.set_text("Cleaning up batch backends...")
            self._cleanup_gpu_backends()

        # Ensure model is loaded and optimized if needed
        if not self._ensure_model_ready("outpaint"):
            return

        self._do_generate_outpaint()

    def _do_generate_outpaint(self):
        """Perform the actual outpaint generation (called after models are loaded)."""
        # For outpaint, always use the current displayed image (not the stored original)
        # This allows chaining multiple outpaints on the progressively larger image
        current_image = self._image_display.get_pil_image()

        extensions = self._image_display.get_outpaint_extensions()
        positive = self._prompt_section.get_positive_prompt()
        negative = self._prompt_section.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)
        strength = self._params_widget.get_strength()

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

        # Set up progress widget for single image (0/1 at start, 1/1 when complete)
        self._progress_widget.set_batch_progress(0, 1)
        self._progress_widget.set_status("Starting outpaint...")
        self._progress_widget.set_generating(True)

        # Get upscale settings
        upscale_enabled = self._upscale_widget.is_enabled
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        # Get output directory from gallery
        output_dir = self._thumbnail_gallery.get_output_directory()

        # Get active LoRAs
        loras = self._get_active_loras()

        generation_service.generate_outpaint(
            params,
            input_image=current_image,
            extensions=extensions,
            strength=strength,
            upscale_enabled=upscale_enabled,
            upscale_model_path=upscale_model_path,
            upscale_model_name=upscale_model_name,
            output_dir=output_dir,
            loras=loras if loras else None,
        )

    # Crop mode handlers
    def _on_crop_mode_changed(self, enabled: bool):
        """Handle Crop Mode toggle."""
        self._image_display.set_crop_mode(enabled)

        # Set callback for crop mask changes
        if enabled:
            self._image_display.set_on_crop_mask_changed(self._on_crop_mask_exists_changed)
        else:
            self._image_display.set_on_crop_mask_changed(None)

        # Update center panel visual indication
        center_panel = self._image_display.get_parent()
        if center_panel:
            if enabled:
                center_panel.add_css_class("center-panel-crop-mode")
                center_panel.remove_css_class("center-panel-edit-mode")
                center_panel.remove_css_class("center-panel-outpaint-mode")
            else:
                center_panel.remove_css_class("center-panel-crop-mode")

        self._status_bar.set_text("Crop mode enabled" if enabled else "Crop mode disabled")

    def _on_crop_mask_exists_changed(self, exists: bool):
        """Handle crop mask creation/deletion."""
        self._toolbar.set_crop_mask_exists(exists)

    def _on_crop_tool_changed(self, tool):
        """Handle crop tool change."""
        from src.ui.widgets.image_display import CropTool as ImageCropTool
        if tool.value == "draw":
            self._image_display.set_crop_tool(ImageCropTool.DRAW)
        else:
            self._image_display.set_crop_tool(ImageCropTool.NONE)

    def _on_clear_crop_mask(self):
        """Handle Clear Crop Mask button click."""
        self._image_display.clear_crop_mask()
        self._status_bar.set_text("Crop mask cleared")

    def _on_crop_image(self):
        """Handle Crop Image button click."""
        if not self._image_display.has_crop_mask():
            self._status_bar.set_text("Please draw a crop region first")
            return

        # Get the cropped image
        cropped = self._image_display.crop_image()
        if cropped is None:
            self._status_bar.set_text("Crop failed - no valid region")
            return

        # Clear the crop mask
        self._image_display.clear_crop_mask()

        # Display the cropped image
        self._image_display.set_image(cropped)

        # Save the cropped image
        output_dir = self._thumbnail_gallery.get_output_directory()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crop_{timestamp}.png"
        filepath = output_dir / filename

        try:
            cropped.save(filepath)
            self._thumbnail_gallery.add_image(filepath, cropped)
            self._status_bar.set_text(f"Cropped: {filename} ({cropped.width}x{cropped.height})")
        except Exception as e:
            self._status_bar.set_text(f"Error saving cropped image: {e}")

        # Update toolbar has_image state
        self._toolbar.set_has_image(True)
        self._update_upscale_button_state()

    def _on_crop_size_changed(self, width: int, height: int):
        """Handle crop size selection from dropdown."""
        if not self._image_display.has_image():
            self._status_bar.set_text("No image loaded")
            return

        self._image_display.set_crop_size(width, height)
        self._status_bar.set_text(f"Crop mask set to {width}x{height}")

    def _on_remove_with_mask(self):
        """Handle Remove with Mask button click."""
        if not self._image_display.has_crop_mask():
            self._status_bar.set_text("Please draw a crop region first")
            return

        # Get the image with the masked area removed and filled
        result = self._image_display.remove_with_mask()
        if result is None:
            self._status_bar.set_text("Remove with mask failed - no valid region")
            return

        # Clear the crop mask
        self._image_display.clear_crop_mask()

        # Display the modified image
        self._image_display.set_image(result)

        # Save the modified image
        output_dir = self._thumbnail_gallery.get_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"removed_{timestamp}.png"
        filepath = output_dir / filename

        try:
            result.save(filepath)
            self._thumbnail_gallery.add_image(filepath, result)
            self._status_bar.set_text(f"Removed region: {filename}")
        except Exception as e:
            self._status_bar.set_text(f"Error saving image: {e}")

    # Refiner mode handlers
    def _on_refiner_mode_changed(self, enabled: bool):
        """Handle Refiner Mode toggle."""
        self._image_display.set_refiner_mode(enabled)

        # Show/hide refiner prompt
        self._prompt_section.set_refiner_mode(enabled)

        # Update center panel style
        center_panel = self._image_display.get_parent().get_parent()
        if enabled:
            center_panel.add_css_class("center-panel-refiner-mode")
            self._status_bar.set_text("Refiner Mode: Click Detect to find objects")
        else:
            center_panel.remove_css_class("center-panel-refiner-mode")
            self._image_display.clear_refiner_masks()
            self._status_bar.set_text("Refiner Mode disabled")

    def _on_refiner_detect(self):
        """Handle Detect button click - show text prompt dialog."""
        if not self._image_display.has_image():
            self._status_bar.set_text("Please load an image first")
            return

        # Create a dialog to get the text prompt
        dialog = Gtk.Dialog(
            title="Detect Objects",
            transient_for=self.get_root(),
            modal=True,
        )
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Detect", Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        # Content area
        content = dialog.get_content_area()
        content.set_spacing(12)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)

        # Label
        label = Gtk.Label(label="Enter what to detect (e.g., 'face', 'hand', 'person'):")
        label.set_halign(Gtk.Align.START)
        content.append(label)

        # Text entry
        entry = Gtk.Entry()
        entry.set_placeholder_text("face")
        entry.set_text("face")  # Default
        entry.set_activates_default(True)
        content.append(entry)

        # Threshold slider
        threshold_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        threshold_label = Gtk.Label(label="Threshold:")
        threshold_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.1, 1.0, 0.05)
        threshold_scale.set_value(0.5)
        threshold_scale.set_hexpand(True)
        threshold_box.append(threshold_label)
        threshold_box.append(threshold_scale)
        content.append(threshold_box)

        def on_response(dialog, response_id):
            if response_id == Gtk.ResponseType.OK:
                text_prompt = entry.get_text().strip()
                threshold = threshold_scale.get_value()
                if text_prompt:
                    dialog.close()
                    self._run_detection(text_prompt, threshold)
                else:
                    self._status_bar.set_text("Please enter a text prompt")
            else:
                dialog.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _run_detection(self, text_prompt: str, threshold: float = 0.5):
        """Run object detection in background thread."""
        self._status_bar.set_text(f"Detecting '{text_prompt}'...")
        self._toolbar.set_state(GenerationState.LOADING)

        # Set up progress widget for detection
        self._progress_widget.set_batch_progress(0, 1)
        self._progress_widget.set_status("Loading SAM3...")
        self._progress_widget.set_step_fraction(0.0)
        self._progress_widget.set_generating(True)

        image = self._image_display.get_pil_image()
        if image is None:
            self._status_bar.set_text("No image loaded")
            self._toolbar.set_state(GenerationState.IDLE)
            self._progress_widget.reset()
            return

        def update_progress(msg, prog):
            """Update both status bar and progress widget."""
            GLib.idle_add(self._status_bar.set_text, msg)
            GLib.idle_add(self._progress_widget.set_status, msg)
            GLib.idle_add(self._progress_widget.set_step_fraction, prog)

        def detect_thread():
            try:
                # Load model if needed
                if not segmentation_backend.is_loaded:
                    success = segmentation_backend.load_model(
                        progress_callback=update_progress
                    )
                    if not success:
                        GLib.idle_add(self._on_detection_complete, [], "Failed to load segmentation model")
                        return

                # Run detection
                masks = segmentation_backend.detect(
                    image,
                    text_prompt,
                    threshold=threshold,
                    progress_callback=update_progress
                )

                GLib.idle_add(self._on_detection_complete, masks, None)

            except Exception as e:
                import traceback
                traceback.print_exc()
                GLib.idle_add(self._on_detection_complete, [], str(e))

        thread = threading.Thread(target=detect_thread, daemon=True)
        thread.start()

    def _on_detection_complete(self, masks: list[DetectedMask], error: str = None):
        """Handle detection completion."""
        self._toolbar.set_state(GenerationState.IDLE)
        self._progress_widget.set_generating(False)
        self._progress_widget.reset()

        if error:
            self._status_bar.set_text(f"Detection error: {error}")
            return

        if not masks:
            self._status_bar.set_text("No objects detected")
            return

        # Set masks on display
        self._image_display.set_refiner_masks(masks)
        self._toolbar.set_has_refiner_masks(True)
        self._status_bar.set_text(f"Detected {len(masks)} region(s). Click to toggle, double-click to delete.")

    def _on_clear_refiner_masks(self):
        """Handle Clear Refiner Masks button click."""
        self._image_display.clear_refiner_masks()
        self._toolbar.set_has_refiner_masks(False)
        self._status_bar.set_text("Refiner masks cleared")

    def _on_generate_refine(self):
        """Handle Generate Refine button click."""
        # Validate we have selected masks
        selected_masks = self._image_display.get_selected_refiner_masks()
        if not selected_masks:
            self._status_bar.set_text("Please select at least one mask to refine")
            return

        # Get refiner prompt (uses the dedicated refiner prompt field)
        positive = self._prompt_section.get_refiner_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a refinement prompt")
            return

        # Check if a checkpoint is selected
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
            return

        # Ensure model is loaded
        if not self._ensure_model_ready("refine"):
            return

        self._do_generate_refine(selected_masks)

    def _do_generate_refine(self, masks: list[DetectedMask]):
        """Perform the actual refinement generation."""
        # Use refiner prompt for generation, pass original prompt separately for metadata
        refiner_prompt = self._prompt_section.get_refiner_prompt()
        original_prompt = self._prompt_section.get_positive_prompt()
        negative = self._prompt_section.get_negative_prompt()

        # Use refiner prompt for generation params
        params = self._params_widget.get_params(refiner_prompt, negative)
        strength = self._params_widget.get_strength()

        input_image = self._image_display.get_pil_image()
        if input_image is None:
            self._status_bar.set_text("No image loaded")
            return

        # Unload segmentation model to free VRAM
        if segmentation_backend.is_loaded:
            segmentation_backend.unload_model()
            self._status_bar.set_text("Unloaded segmentation model for generation...")

        # Set up progress widget
        self._progress_widget.set_batch_progress(0, len(masks))
        self._progress_widget.set_status("Starting refinement...")
        self._progress_widget.set_generating(True)

        # Get upscale settings
        upscale_model_path = self._upscale_widget.selected_model_path
        upscale_model_name = self._upscale_widget.selected_model_name

        # Get output directory from gallery
        output_dir = self._thumbnail_gallery.get_output_directory()

        # Get active LoRAs
        loras = self._get_active_loras()

        generation_service.generate_refine(
            params=params,
            input_image=input_image,
            masks=masks,
            strength=strength,
            upscale_model_path=upscale_model_path,
            output_dir=output_dir,
            loras=loras if loras else None,
            original_prompt=original_prompt,
        )

        # Update toolbar has_image state
        self._toolbar.set_has_image(True)
        self._update_upscale_button_state()

    def _on_state_changed(self, state: GenerationState):
        """Handle generation state change."""
        self._toolbar.set_state(state)

        if state == GenerationState.IDLE:
            self._toolbar.set_model_loaded(generation_service.is_model_loaded)

            # Check for pending generation after model loading
            if self._pending_generation:
                pending = self._pending_generation
                self._pending_generation = None
                # Only execute if models loaded successfully
                if generation_service.is_model_loaded:
                    if pending == "generate":
                        self._do_generate()
                    elif pending == "img2img":
                        self._do_img2img()
                    elif pending == "inpaint":
                        self._do_generate_inpaint()
                    elif pending == "outpaint":
                        self._do_generate_outpaint()
                    elif pending == "refine":
                        selected_masks = self._image_display.get_selected_refiner_masks()
                        if selected_masks:
                            self._do_generate_refine(selected_masks)
                        else:
                            self._status_bar.set_text("No masks selected for refinement")
                else:
                    self._status_bar.set_text("Model loading failed - generation cancelled")

    def _on_progress(self, message: str, progress: float):
        """Handle progress update."""
        self._status_bar.set_text(message)
        self._progress_widget.set_status(message)
        self._progress_widget.set_step_fraction(progress)

    def _on_step_progress(self, step: int, total: int):
        """Handle step progress update."""
        self._progress_widget.set_step_progress(step, total)
        self._progress_widget.set_status(f"Step {step}/{total}")

    def _on_generation_complete(self, result: GenerationResult):
        """Handle generation completion (for single-image generation via generation_service)."""
        # Update progress to show completion
        self._progress_widget.set_batch_progress(1, 1)
        self._progress_widget.set_step_fraction(1.0)
        self._progress_widget.set_generating(False)

        if result.success and result.image:
            self._progress_widget.set_status("Complete")

            # In inpaint mode, we want to keep the mask and show the result
            # The user can continue to refine with the same mask
            in_inpaint_mode = self._image_display.inpaint_mode
            in_outpaint_mode = self._image_display.outpaint_mode
            in_refiner_mode = self._image_display.refiner_mode

            # For outpaint mode, clear masks before setting new image
            # since the new image has different dimensions
            if in_outpaint_mode:
                self._image_display.clear_outpaint_masks()

            # Display the generated image
            self._image_display.set_image(result.image)

            # Add to thumbnail gallery
            if result.path:
                self._thumbnail_gallery.add_image(result.path, result.image)

            # Only update seed widget if user specified a seed (not random)
            # This preserves -1 for random seed behavior
            # Skip seed update for refiner mode (each mask uses different seeds)
            if not self._last_seed_was_random and result.seed != -1 and not in_refiner_mode:
                self._params_widget.set_seed(result.seed)

            # Show the actual seed used in status bar
            seed_info = f" (seed: {result.seed})" if result.seed != -1 else ""
            mode_info = " (inpaint)" if in_inpaint_mode else (" (outpaint)" if in_outpaint_mode else (" (refine)" if in_refiner_mode else ""))
            self._status_bar.set_text(f"Generated: {result.path.name if result.path else 'image'}{seed_info}{mode_info}")

            # Update toolbar has_image state
            self._toolbar.set_has_image(True)
            self._update_upscale_button_state()
        else:
            error_msg = f"Generation failed: {result.error or 'Unknown error'}"
            self._status_bar.set_text(error_msg)
            self._progress_widget.set_status("Failed")

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
            self._prompt_section.set_prompts(
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

        # Update upscale button state
        self._update_upscale_button_state()

    def _on_image_dropped(self, source_path: Path):
        """Handle image dropped onto the image display area."""
        import shutil
        from PIL import Image

        # Get the target directory
        output_dir = self._thumbnail_gallery.get_output_directory()

        # Ensure the directory exists
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Determine the target filename (rename if exists)
        target_path = output_dir / source_path.name
        if target_path.exists():
            # Find a unique name by adding a counter
            base_name = source_path.stem
            extension = source_path.suffix
            counter = 1
            while target_path.exists():
                target_path = output_dir / f"{base_name}_{counter}{extension}"
                counter += 1

        try:
            # Copy the file to the target directory
            shutil.copy2(source_path, target_path)

            # Load the image
            image = Image.open(target_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Display the image
            self._image_display.set_image(image)

            # Add to the gallery
            self._thumbnail_gallery.add_image(target_path, image)

            # Update toolbar state
            self._toolbar.set_has_image(True)
            self._update_upscale_button_state()

            # Try to load metadata if available
            metadata = load_metadata_from_image(target_path)
            if metadata:
                self._prompt_section.set_prompts(metadata.prompt, metadata.negative_prompt)
                self._params_widget.set_size(metadata.width, metadata.height)
                self._params_widget.set_steps(metadata.steps)
                self._params_widget.set_cfg_scale(metadata.cfg_scale)
                self._params_widget.set_seed(metadata.seed)
                self._params_widget.set_sampler(metadata.sampler)
                self._status_bar.set_text(f"Imported: {target_path.name} (parameters restored)")
            else:
                self._status_bar.set_text(f"Imported: {target_path.name}")

        except Exception as e:
            self._status_bar.set_text(f"Error importing image: {e}")

    def _on_upscale_settings_changed(self):
        """Handle upscale settings change (enabled/model changed)."""
        self._update_upscale_button_state()

    def _on_batch_settings_changed(self):
        """Handle batch settings change."""
        # Batch mode affects generation behavior but doesn't need immediate UI updates
        pass

    def _update_upscale_button_state(self):
        """Update the upscale button state based on current conditions."""
        # Guard against being called before UI is fully built
        if not hasattr(self, '_image_display') or not hasattr(self, '_toolbar'):
            return
        # Check if there's an image to upscale
        has_image = self._image_display.get_pil_image() is not None
        # Check if upscaling is enabled and a model is selected
        upscale_enabled = self._upscale_widget.is_enabled and self._upscale_widget.selected_model is not None
        self._toolbar.set_upscale_enabled(upscale_enabled, has_image)

    def _on_upscale(self):
        """Handle Upscale button click."""
        # Get the current image
        current_image = self._image_display.get_pil_image()
        if current_image is None:
            self._status_bar.set_text("No image to upscale")
            return

        # Check if upscaling is enabled
        if not self._upscale_widget.is_enabled:
            self._status_bar.set_text("Enable upscaling first")
            return

        # Get the upscale model path
        upscale_model_path = self._upscale_widget.selected_model_path
        if not upscale_model_path:
            self._status_bar.set_text("Select an upscale model first")
            return

        upscale_model_name = self._upscale_widget.selected_model_name
        self._status_bar.set_text(f"Upscaling with {upscale_model_name}...")

        def update_upscale_progress(msg, prog):
            """Helper to update progress widget."""
            self._progress_widget.set_status(msg)
            self._progress_widget.set_step_fraction(prog)

        # Run upscaling in background thread
        def upscale_thread():
            try:
                # Load the upscale model if not already loaded or different model
                if upscale_backend._loaded_model_path != upscale_model_path:
                    GLib.idle_add(lambda: update_upscale_progress("Loading upscale model...", 0.1))
                    success = upscale_backend.load_model(
                        upscale_model_path,
                        progress_callback=lambda msg, prog: GLib.idle_add(
                            lambda m=msg, p=prog: update_upscale_progress(m, p)
                        )
                    )
                    if not success:
                        GLib.idle_add(lambda: self._on_upscale_complete(None, "Failed to load upscale model"))
                        return

                # Perform upscaling
                GLib.idle_add(lambda: update_upscale_progress("Upscaling image...", 0.5))
                upscaled_image = upscale_backend.upscale(
                    current_image,
                    progress_callback=lambda msg, prog: GLib.idle_add(
                        lambda m=msg, p=prog: update_upscale_progress(m, 0.5 + prog * 0.4)
                    )
                )

                if upscaled_image:
                    # Save the upscaled image
                    output_dir = self._thumbnail_gallery.get_output_directory()
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Generate filename
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"upscaled_{timestamp}.png"
                    output_path = output_dir / filename

                    GLib.idle_add(lambda: update_upscale_progress("Saving image...", 0.95))
                    upscaled_image.save(output_path, "PNG")
                    GLib.idle_add(lambda: self._on_upscale_complete(upscaled_image, None, output_path))
                else:
                    GLib.idle_add(lambda: self._on_upscale_complete(None, "Upscaling failed"))

            except Exception as e:
                print(f"Error during upscaling: {e}")
                import traceback
                traceback.print_exc()
                GLib.idle_add(lambda: self._on_upscale_complete(None, str(e)))

        # Set UI to generating state
        self._toolbar.set_state(GenerationState.GENERATING)
        self._progress_widget.set_batch_progress(1, 1)
        self._progress_widget.set_status("Starting upscale...")
        self._progress_widget.set_generating(True)

        thread = threading.Thread(target=upscale_thread, daemon=True)
        thread.start()

    def _on_upscale_complete(self, image, error, path=None):
        """Handle upscale completion (called from main thread)."""
        self._toolbar.set_state(GenerationState.IDLE)
        self._progress_widget.reset()
        self._progress_widget.set_generating(False)

        if image and not error:
            # Display the upscaled image
            self._image_display.set_image(image)

            # Add to thumbnail gallery
            if path:
                self._thumbnail_gallery.add_image(path, image)

            scale = upscale_backend.scale
            self._status_bar.set_text(f"Upscaled {scale}x: {path.name if path else 'image'}")

            # Update toolbar has_image state
            self._toolbar.set_has_image(True)
            self._update_upscale_button_state()
        else:
            self._status_bar.set_text(f"Upscale failed: {error or 'Unknown error'}")

    def get_panel_positions(self) -> dict:
        """Get current panel positions for saving to config."""
        # For collapsed panels, use the saved (expanded) position instead of current
        left_pos = (self._left_collapse_btn._saved_position
                    if self._left_collapse_btn.collapsed and self._left_collapse_btn._saved_position
                    else self._paned_outer.get_position())
        right_pos = (self._right_collapse_btn._saved_position
                     if self._right_collapse_btn.collapsed and self._right_collapse_btn._saved_position
                     else self._paned_inner.get_position())
        center_pos = (self._prompt_collapse_btn._saved_position
                      if self._prompt_collapse_btn.collapsed and self._prompt_collapse_btn._saved_position
                      else self._center_paned.get_position())

        positions = {
            "left": left_pos,
            "right": right_pos,
            "center": center_pos,
        }
        # Add prompt section positions
        prompt_positions = self._prompt_section.get_paned_positions()
        positions["prompt_section_width"] = prompt_positions.get("prompts_width", -1)
        positions["prompt_section_split"] = prompt_positions.get("prompts_split", -1)
        # Add prompt manager internal positions
        positions["prompt_manager_split"] = self._prompt_section.prompt_manager.get_paned_position()
        # Add prompt font sizes
        font_sizes = self._prompt_section.get_font_sizes()
        positions["positive_prompt_font_size"] = font_sizes.get("positive", 0)
        positions["negative_prompt_font_size"] = font_sizes.get("negative", 0)
        positions["refiner_prompt_font_size"] = font_sizes.get("refiner", 0)
        # Add collapsed states
        positions["left_panel_collapsed"] = self._left_collapse_btn.collapsed
        positions["right_panel_collapsed"] = self._right_collapse_btn.collapsed
        positions["prompt_panel_collapsed"] = self._prompt_collapse_btn.collapsed
        return positions
