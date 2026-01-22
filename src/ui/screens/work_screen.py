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
from src.ui.widgets.prompt_entry import PromptPanel
from src.ui.widgets.image_display import ImageDisplayFrame
from src.ui.widgets.thumbnail_gallery import ThumbnailGallery
from src.ui.widgets.toolbar import Toolbar
from src.ui.widgets.upscale_settings import UpscaleSettingsWidget
from src.ui.widgets.batch_settings import BatchSettingsWidget
from src.ui.widgets.toolbar import InpaintTool
from src.ui.widgets.image_display import MaskTool
from src.ui.widgets.lora_selector import LoRASelectorPanel
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

        # Restore panel sizes from config
        window_config = config_manager.config.window
        self._paned_outer.set_position(window_config.left_panel_width)
        self._paned_inner.set_position(window_config.right_panel_position)

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

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(8)
        box.set_margin_end(8)
        scrolled.set_child(box)

        # VRAM display (at the top)
        self._vram_display = VRAMDisplay()
        box.append(self._vram_display)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(6)
        separator.set_margin_bottom(6)
        box.append(separator)

        # Model selectors section
        model_header = Gtk.Label(label="Models")
        model_header.add_css_class("section-header")
        model_header.set_halign(Gtk.Align.START)
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
        self._optimize_checkbox.set_tooltip_text(
            "When enabled, uses torch.compile for faster generation. "
            "First generation will take longer to compile, but subsequent generations will be much faster. "
            "Compiled kernels are cached and reused across sessions."
        )
        self._optimize_checkbox.add_css_class("caption")
        self._optimize_checkbox.set_active(False)  # Default to disabled
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

        # Restore center panel position from config
        self._center_paned.set_position(config_manager.config.window.center_panel_height)

        return self._center_paned

    def _create_right_panel(self) -> Gtk.Widget:
        """Create the right panel with thumbnail gallery."""
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.add_css_class("right-panel")
        scrolled.set_size_request(150, -1)  # Minimum width

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
            self._prompt_panel.set_prompts("", "")
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
        self._toolbar.set_progress(message, progress)

    def _on_optimization_complete(self, success: bool, error: str = None):
        """Handle optimization completion."""
        self._toolbar.clear_progress()
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
        else:
            self._pending_generation = None
            self._status_bar.set_text(f"Optimization failed: {error or 'Unknown error'}")

    def _on_generate(self):
        """Handle Generate button click."""
        # Check if a checkpoint is selected
        if not model_manager.loaded.checkpoint:
            self._status_bar.set_text("Please select a checkpoint first")
            return

        positive = self._prompt_panel.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        # For batch mode, skip normal model loading - batch handles its own
        batch_count = self._batch_widget.batch_count
        if batch_count > 1:
            self._start_batch_generation("generate")
            return

        # Ensure model is loaded and optimized if needed (single image only)
        if not self._ensure_model_ready("generate"):
            return

        self._do_generate()

    def _do_generate(self):
        """Perform the actual generation (called after models are loaded, single image only)."""
        positive = self._prompt_panel.get_positive_prompt()
        negative = self._prompt_panel.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

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

        # Store input image for img2img batch (use the same image for all)
        if batch_type == "img2img":
            self._batch_input_image = self._image_display.get_pil_image()

        # Get selected GPUs
        gpu_indices = self._batch_widget.selected_gpu_indices
        if not gpu_indices:
            gpu_indices = [0]  # Default to GPU 0

        self._status_bar.set_text(f"Starting batch generation: {self._batch_count} images on {len(gpu_indices)} GPU(s)...")

        # Disable batch widget and toolbar during generation
        self._batch_widget.set_sensitive_all(False)
        self._toolbar.set_state(GenerationState.GENERATING)

        # Start batch generation in a background thread
        thread = threading.Thread(
            target=self._run_parallel_batch,
            args=(batch_type, gpu_indices),
            daemon=True
        )
        thread.start()

    def _run_parallel_batch(self, batch_type: str, gpu_indices: list[int]):
        """Run parallel batch generation across multiple GPUs."""
        import torch
        from PIL import Image

        try:
            # Get generation parameters
            positive = self._prompt_panel.get_positive_prompt()
            negative = self._prompt_panel.get_negative_prompt()
            params = self._params_widget.get_params(positive, negative)
            strength = self._params_widget.get_strength()

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
            # Note: torch.compile with CUDA graphs doesn't work well across multiple GPUs
            # due to CUDA graph capture conflicts, so we only enable it for single-GPU batch
            optimize_enabled = self._optimize_checkbox.get_active()
            use_compiled = False

            if optimize_enabled and len(gpu_indices) > 1:
                # Multi-GPU batch with optimization requested - warn user and disable
                GLib.idle_add(
                    self._status_bar.set_text,
                    f"Note: Optimization disabled for multi-GPU batch (CUDA graph conflicts). Loading normally..."
                )
                optimize_enabled = False  # Disable for this batch

            if optimize_enabled:
                # Single GPU batch - optimization is safe to use
                # Check if compiled cache exists
                has_cache = diffusers_backend.has_compiled_cache(
                    checkpoint_path, params.width, params.height
                )

                if has_cache:
                    GLib.idle_add(
                        self._status_bar.set_text,
                        f"Compiled cache found. Loading optimized model on GPU {gpu_indices[0]}..."
                    )
                    use_compiled = True
                else:
                    # Need to compile first
                    GLib.idle_add(
                        self._status_bar.set_text,
                        f"Compiling model on GPU {gpu_indices[0]} (first time only)..."
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

                    if not success:
                        GLib.idle_add(
                            self._status_bar.set_text,
                            "Compilation failed. Proceeding without optimization..."
                        )
                        use_compiled = False
                    else:
                        GLib.idle_add(
                            self._status_bar.set_text,
                            f"Compilation complete. Loading optimized model on GPU {gpu_indices[0]}..."
                        )
                        use_compiled = True

            if not optimize_enabled and len(gpu_indices) == 1:
                GLib.idle_add(self._status_bar.set_text, f"Loading model on GPU {gpu_indices[0]}...")
            elif not optimize_enabled and len(gpu_indices) > 1:
                GLib.idle_add(self._status_bar.set_text, f"Loading models on {len(gpu_indices)} GPU(s)...")

            # Create backend instances for each GPU and load models
            self._gpu_backends = {}
            for i, gpu_idx in enumerate(gpu_indices):
                if self._batch_cancelled:
                    break

                GLib.idle_add(
                    self._status_bar.set_text,
                    f"Loading {'optimized ' if use_compiled else ''}model on GPU {gpu_idx} ({i + 1}/{len(gpu_indices)})..."
                )

                backend = DiffusersBackend(gpu_index=gpu_idx)
                success = backend.load_model(
                    checkpoint_path=checkpoint_path,
                    model_type=model_type,
                    vae_path=vae_path,
                    use_compiled=use_compiled,
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
            completed = 0
            gpu_list = list(self._gpu_backends.keys())

            # Use ThreadPoolExecutor to run generations in parallel
            with ThreadPoolExecutor(max_workers=len(self._gpu_backends)) as executor:
                futures = {}

                while remaining > 0 or futures:
                    if self._batch_cancelled:
                        # Cancel pending futures
                        for future in futures:
                            future.cancel()
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

                        # Create params with random seed for this generation
                        gen_params = GenerationParams(
                            prompt=params.prompt,
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
                                upscale_model_name, checkpoint_name, vae_name, model_type
                            )
                        else:  # img2img
                            future = executor.submit(
                                self._generate_img2img_on_gpu,
                                backend, gen_params, self._batch_input_image, strength,
                                output_dir, upscale_enabled, upscale_model_path,
                                upscale_model_name, checkpoint_name, vae_name, model_type
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

                                # Update UI with result
                                if result_path and result_image:
                                    GLib.idle_add(
                                        self._on_batch_image_complete,
                                        result_path, result_image, completed, self._batch_count
                                    )
                            except Exception as e:
                                print(f"Generation error on GPU {gpu_idx}: {e}")
                                completed += 1

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
            # Clean up GPU backends
            self._cleanup_gpu_backends()
            GLib.idle_add(self._end_batch_generation)

    def _generate_on_gpu(self, backend: DiffusersBackend, params: GenerationParams,
                         output_dir: Path, upscale_enabled: bool, upscale_model_path: str,
                         upscale_model_name: str, checkpoint_name: str, vae_name: str,
                         model_type: str):
        """Generate a single image on a specific GPU backend."""
        from src.utils.metadata import save_image_with_metadata, GenerationMetadata

        image = backend.generate(params)
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

        save_image_with_metadata(image, output_path, metadata)

        return output_path, image

    def _generate_img2img_on_gpu(self, backend: DiffusersBackend, params: GenerationParams,
                                  input_image, strength: float, output_dir: Path,
                                  upscale_enabled: bool, upscale_model_path: str,
                                  upscale_model_name: str, checkpoint_name: str, vae_name: str,
                                  model_type: str):
        """Generate a single img2img image on a specific GPU backend."""
        from src.utils.metadata import save_image_with_metadata, GenerationMetadata

        image = backend.generate_img2img(params, input_image, strength)
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

        return output_path, image

    def _on_batch_image_complete(self, path: Path, image, completed: int, total: int):
        """Handle completion of a single batch image (called on main thread)."""
        # Display the latest image
        self._image_display.set_image(image)

        # Add to thumbnail gallery
        self._thumbnail_gallery.add_image(path, image)

        # Update status
        self._status_bar.set_text(f"Batch progress: {completed}/{total} - {path.name}")

        # Update toolbar
        self._toolbar.set_has_image(True)
        self._update_upscale_button_state()

    def _on_batch_compile_progress(self, message: str, progress: float):
        """Handle compilation progress during batch setup (called on main thread)."""
        self._status_bar.set_text(message)
        self._toolbar.set_progress(message, progress)

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
        count = self._batch_count
        self._batch_mode = False
        self._batch_type = None
        self._batch_count = 0
        self._batch_current = 0
        self._batch_input_image = None
        self._batch_cancelled = False

        # Re-enable batch widget and toolbar
        self._batch_widget.set_sensitive_all(True)
        self._toolbar.set_state(GenerationState.IDLE)

        self._status_bar.set_text(f"Batch generation complete: {count} images generated")

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

        positive = self._prompt_panel.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

        # For batch mode, skip normal model loading - batch handles its own
        batch_count = self._batch_widget.batch_count
        if batch_count > 1:
            self._start_batch_generation("img2img")
            return

        # Ensure model is loaded and optimized if needed (single image only)
        if not self._ensure_model_ready("img2img"):
            return

        self._do_img2img()

    def _do_img2img(self):
        """Perform the actual img2img generation (called after models are loaded, single image only)."""
        input_image = self._image_display.get_pil_image()

        positive = self._prompt_panel.get_positive_prompt()
        negative = self._prompt_panel.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)
        strength = self._params_widget.get_strength()

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

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

        positive = self._prompt_panel.get_positive_prompt()
        if not positive.strip():
            self._status_bar.set_text("Please enter a positive prompt")
            return

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
        positive = self._prompt_panel.get_positive_prompt()
        negative = self._prompt_panel.get_negative_prompt()
        params = self._params_widget.get_params(positive, negative)
        strength = self._params_widget.get_strength()

        # Store whether user wanted random seed
        self._last_seed_was_random = (params.seed == -1)

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
                else:
                    self._status_bar.set_text("Model loading failed - generation cancelled")

    def _on_progress(self, message: str, progress: float):
        """Handle progress update."""
        self._toolbar.set_progress(message, progress)
        self._status_bar.set_text(message)

    def _on_step_progress(self, step: int, total: int):
        """Handle step progress update."""
        self._toolbar.set_step_progress(step, total)

    def _on_generation_complete(self, result: GenerationResult):
        """Handle generation completion (for single-image generation via generation_service)."""
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
            self._update_upscale_button_state()
        else:
            error_msg = f"Generation failed: {result.error or 'Unknown error'}"
            self._status_bar.set_text(error_msg)

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

        # Update upscale button state
        self._update_upscale_button_state()

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

        # Run upscaling in background thread
        def upscale_thread():
            try:
                # Load the upscale model if not already loaded or different model
                if upscale_backend._loaded_model_path != upscale_model_path:
                    GLib.idle_add(lambda: self._toolbar.set_progress("Loading upscale model...", 0.1))
                    success = upscale_backend.load_model(
                        upscale_model_path,
                        progress_callback=lambda msg, prog: GLib.idle_add(
                            lambda m=msg, p=prog: self._toolbar.set_progress(m, p)
                        )
                    )
                    if not success:
                        GLib.idle_add(lambda: self._on_upscale_complete(None, "Failed to load upscale model"))
                        return

                # Perform upscaling
                GLib.idle_add(lambda: self._toolbar.set_progress("Upscaling image...", 0.5))
                upscaled_image = upscale_backend.upscale(
                    current_image,
                    progress_callback=lambda msg, prog: GLib.idle_add(
                        lambda m=msg, p=prog: self._toolbar.set_progress(m, 0.5 + prog * 0.4)
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
        self._toolbar.set_progress("Starting upscale...", 0.0)

        thread = threading.Thread(target=upscale_thread, daemon=True)
        thread.start()

    def _on_upscale_complete(self, image, error, path=None):
        """Handle upscale completion (called from main thread)."""
        self._toolbar.set_state(GenerationState.IDLE)
        self._toolbar.clear_progress()

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
        return {
            "left": self._paned_outer.get_position(),
            "right": self._paned_inner.get_position(),
            "center": self._center_paned.get_position(),
        }
