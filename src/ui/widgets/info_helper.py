"""Information helper widgets for tooltips and info popovers."""

from typing import Optional

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib


class InfoButton(Gtk.MenuButton):
    """Small info button that shows a popover with explanation text when clicked."""

    def __init__(self, info_text: str):
        super().__init__()
        self._info_text = info_text

        # Set up the button appearance - use "i" in circle icon
        self.set_icon_name("help-about-symbolic")
        self.add_css_class("flat")
        self.add_css_class("info-button")
        self.set_valign(Gtk.Align.CENTER)
        self.set_has_frame(False)

        # Create popover with info text
        self._popover = Gtk.Popover()
        self._popover.add_css_class("info-popover")

        # Create label for info text
        label = Gtk.Label(label=info_text)
        label.set_wrap(True)
        label.set_max_width_chars(50)
        label.set_margin_top(12)
        label.set_margin_bottom(12)
        label.set_margin_start(12)
        label.set_margin_end(12)
        label.set_xalign(0)
        label.add_css_class("info-popover-text")

        self._popover.set_child(label)
        self.set_popover(self._popover)


class SectionHeader(Gtk.Box):
    """Section header with label and optional info button."""

    def __init__(self, label: str, info_text: Optional[str] = None):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Label - vertically centered
        self._label = Gtk.Label(label=label)
        self._label.add_css_class("section-header")
        self._label.set_halign(Gtk.Align.START)
        self._label.set_valign(Gtk.Align.CENTER)
        self.append(self._label)

        # Info button (if info text provided) - vertically centered
        if info_text:
            self._info_button = InfoButton(info_text)
            self._info_button.set_valign(Gtk.Align.CENTER)
            self.append(self._info_button)

        self.set_halign(Gtk.Align.START)
        self.set_valign(Gtk.Align.CENTER)


def add_hover_tooltip(widget: Gtk.Widget, tooltip_text: str, delay_ms: int = 1000):
    """Add a tooltip to a widget that appears after hovering for a specified delay.

    Args:
        widget: The GTK widget to add the tooltip to
        tooltip_text: The text to show in the tooltip
        delay_ms: Delay in milliseconds before showing tooltip (default 1000ms)
    """
    # Store state on the widget
    widget._tooltip_timeout_id = None
    widget._tooltip_text = tooltip_text

    def on_enter(controller, x, y):
        # Cancel any existing timeout
        if widget._tooltip_timeout_id:
            GLib.source_remove(widget._tooltip_timeout_id)

        # Set up delayed tooltip
        def show_tooltip():
            widget.set_tooltip_text(widget._tooltip_text)
            widget._tooltip_timeout_id = None
            return False  # Don't repeat

        widget._tooltip_timeout_id = GLib.timeout_add(delay_ms, show_tooltip)

    def on_leave(controller):
        # Cancel pending tooltip
        if widget._tooltip_timeout_id:
            GLib.source_remove(widget._tooltip_timeout_id)
            widget._tooltip_timeout_id = None
        # Clear tooltip when leaving
        widget.set_tooltip_text(None)

    # Add motion controller for enter/leave events
    motion = Gtk.EventControllerMotion()
    motion.connect("enter", on_enter)
    motion.connect("leave", on_leave)
    widget.add_controller(motion)


# Section info texts
SECTION_INFO = {
    "monitoring": (
        "GPU Monitoring\n\n"
        "Displays real-time GPU status:\n"
        "• VRAM bar: Shows memory usage (green/yellow/red)\n"
        "• Usage indicator: Small box showing GPU utilization %\n"
        "• Temperature: Current GPU die temperature in °C\n\n"
        "Monitoring runs in a background thread."
    ),
    "models": (
        "Model Selection\n\n"
        "Select the AI models for image generation:\n"
        "• Checkpoint: Main Stable Diffusion model (required)\n"
        "• VAE: Variational Auto-Encoder for image decoding\n"
        "• CLIP: Text encoder (usually embedded in checkpoint)\n\n"
        "• Optimize: Enable torch.compile for faster generation.\n"
        "  WARNING: Takes very long on first run (many minutes).\n"
        "  Only recommended for 50+ batch images on single GPU.\n"
        "  Fixed resolution per optimization.\n"
        "  Does NOT work with Inpaint/Outpaint modes.\n"
        "  Does NOT work with multi-GPU batch generation."
    ),
    "parameters": (
        "Generation Parameters\n\n"
        "Configure how images are generated:\n"
        "• Size: Output image dimensions (width × height)\n"
        "• Steps: Number of denoising steps (more = higher quality)\n"
        "• CFG Scale: How closely to follow the prompt (7-12 typical)\n"
        "• Seed: Random seed (-1 for random each time)\n"
        "• Sampler: Denoising algorithm (Euler, DPM++, etc.)\n"
        "• Scheduler: Noise schedule type\n"
        "• Strength: For img2img/inpaint (0-1, lower = more original)"
    ),
    "lora": (
        "LoRA Selection\n\n"
        "LoRAs (Low-Rank Adaptations) are small model add-ons:\n"
        "• Add styles, characters, or concepts to generations\n"
        "• Multiple LoRAs can be combined\n"
        "• Weight slider controls influence strength (0.0-2.0)\n\n"
        "Place LoRA files in the configured loras directory."
    ),
    "upscale": (
        "Upscale Settings\n\n"
        "Enhance image resolution after generation:\n"
        "• Enable: Toggle automatic upscaling\n"
        "• Model: Select upscaling model (RealESRGAN, etc.)\n\n"
        "Upscaling increases image size by 4x (e.g., 1024→4096).\n"
        "Can also upscale existing images with the Upscale button."
    ),
    "batch": (
        "Batch Generation\n\n"
        "Generate multiple images in parallel:\n"
        "• Count: Number of images to generate\n"
        "• GPUs: Select which GPUs to use\n\n"
        "With multiple GPUs, images generate in parallel.\n"
        "Each image uses a random seed for variety."
    ),
    "gallery": (
        "Image Gallery\n\n"
        "Browse and manage generated images:\n"
        "• Click thumbnail to load image and restore parameters\n"
        "• Right-click for context menu (delete, open folder)\n"
        "• Use folder dropdown to organize images\n"
        "• Create new subfolders for different projects\n\n"
        "Images are saved with metadata that can be restored."
    ),
    "prompt_management": (
        "Prompt Management\n\n"
        "Create reusable word lists for dynamic prompts:\n"
        "• List: Your saved prompt lists (quality tags, styles, etc.)\n"
        "• Words: Contents of the selected list\n"
        "• Checkbox: Enable/disable a list for generation\n"
        "• Number: How many words to randomly pick (1-10)\n\n"
        "During generation, random words from checked lists\n"
        "are prepended to your positive prompt.\n\n"
        "Each batch image gets a fresh random selection!"
    ),
    "prompt_list": (
        "Prompt Lists\n\n"
        "Manage your word/phrase collections:\n"
        "• Click a list to select it and view its words\n"
        "• Check the box to include it in generation\n"
        "• Set the number to control random picks\n"
        "• + button: Create a new list\n"
        "• - button: Delete the selected list\n\n"
        "Lists are saved as text files in models/prompt-lists/"
    ),
    "prompt_words": (
        "Word List\n\n"
        "Words/phrases in the selected prompt list:\n"
        "• Click a word to select it\n"
        "• + button: Add a new word to the list\n"
        "• - button: Remove the selected word\n\n"
        "Each word should be a tag or short phrase\n"
        "(e.g., 'masterpiece', 'best quality', '8k resolution')"
    ),
    "generation_progress": (
        "Generation Progress\n\n"
        "Shows the current generation status:\n"
        "• Batch: Overall progress for batch generation\n"
        "• Step: Current image progress (steps, VAE, saving)\n"
        "• Step (GPUn): Per-GPU step progress in batch mode\n"
        "• Status: What's currently happening\n\n"
        "Batch progress shows 1/1 for single image generation.\n"
        "In batch mode with multiple GPUs, each GPU shows its own\n"
        "step progress bar."
    ),
}

# Label tooltips
LABEL_TOOLTIPS = {
    # Generation parameters
    "size": "Output image dimensions in pixels (width × height). Common sizes: 1024×1024, 1152×896, etc.",
    "steps": "Number of denoising iterations. More steps = higher quality but slower. Typical: 20-50.",
    "cfg_scale": "Classifier-Free Guidance scale. Higher values follow prompt more strictly. Typical: 7-12.",
    "seed": "Random seed for reproducibility. Use -1 for a new random seed each generation.",
    "sampler": "Denoising algorithm. Popular choices: Euler a, DPM++ 2M Karras, DDIM.",
    "scheduler": "Noise schedule type. Affects how noise is reduced over steps.",
    "strength": "How much to transform the input image. 0.0 = no change, 1.0 = complete redraw.",

    # Model selectors
    "checkpoint": "Main Stable Diffusion model file (.safetensors). This is the primary model for generation.",
    "vae": "Variational Auto-Encoder. Converts latent space to images. 'Embedded' uses the one in checkpoint.",
    "clip": "Text encoder model. Converts prompts to embeddings. Usually embedded in checkpoint.",
    "optimize": (
        "Use torch.compile for optimized generation. "
        "WARNING: This option will take very long to run for the first time (many minutes depending on your GPU). "
        "Unless you want to generate more than 50 images in batch mode using a single GPU, it is not recommended. "
        "Each optimization is for a model with a fixed resolution. "
        "This will not work with Inpainting and Outpainting mode. "
        "It will also not work for multi-GPU batch generation, but it does work for single-GPU batch generation."
    ),

    # Upscale
    "upscale_enable": "Enable automatic upscaling after each generation.",
    "upscale_model": "AI upscaling model to use. RealESRGAN models work well for most images.",

    # Batch
    "batch_count": "Number of images to generate in this batch.",
    "batch_gpus": "GPUs to use for parallel generation. More GPUs = faster batch completion.",

    # LoRA
    "lora_weight": "LoRA influence strength. 1.0 = full effect, 0.5 = half effect, 0.0 = disabled.",

    # Prompt Management
    "prompt_list": "Your saved prompt lists. Check to include in generation, set number for random picks.",
    "prompt_words": "Words in the selected list. These are randomly selected during generation.",
    "prompt_count": "Number of words to randomly pick from this list (1-10). If greater than available words, all are used.",
    "positive_prompt": "Describe what you want in the image. Random words from checked lists are prepended automatically.",
    "negative_prompt": "Describe what you want to avoid. These concepts will be suppressed in generation.",
}
