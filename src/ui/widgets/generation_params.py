"""Generation parameters widget."""

from typing import Callable, Optional

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.utils.constants import (
    SAMPLERS,
    SCHEDULERS,
    SIZE_PRESETS,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    DEFAULT_STEPS,
    DEFAULT_CFG_SCALE,
    DEFAULT_SEED,
    MIN_STEPS,
    MAX_STEPS,
    MIN_CFG_SCALE,
    MAX_CFG_SCALE,
    MIN_SIZE,
    MAX_SIZE,
    SIZE_STEP,
)
from src.core.config import config_manager
from src.backends.diffusers_backend import GenerationParams
from src.ui.widgets.info_helper import SectionHeader, add_hover_tooltip, SECTION_INFO, LABEL_TOOLTIPS


class GenerationParamsWidget(Gtk.Box):
    """Widget for configuring image generation parameters."""

    LABEL_WIDTH = 55  # Fixed label width for alignment

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._build_ui()
        self._load_defaults()

    def _create_label(self, text: str) -> Gtk.Label:
        """Create a compact label with fixed width."""
        label = Gtk.Label(label=text)
        label.set_size_request(self.LABEL_WIDTH, -1)
        label.set_halign(Gtk.Align.START)
        label.add_css_class("caption")
        return label

    def _build_ui(self):
        """Build the parameters UI."""
        # Header with info button
        header = SectionHeader("Generation Parameters", SECTION_INFO["parameters"])
        self.append(header)

        # Size row: Label | Dropdown | W: spin | H: spin
        self.append(self._create_size_section())

        # Sampler
        self.append(self._create_sampler_section())

        # Steps and CFG on same row
        self.append(self._create_steps_cfg_section())

        # Strength (for img2img)
        self.append(self._create_strength_section())

        # Refiner Strength (hidden by default, shown in refiner mode)
        self._refiner_strength_row = self._create_refiner_strength_section()
        self._refiner_strength_row.set_visible(False)
        self.append(self._refiner_strength_row)

        # Seed
        self.append(self._create_seed_section())

    def _create_size_section(self) -> Gtk.Widget:
        """Create size selection section - all on one line."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Size label
        label = self._create_label("Size")
        add_hover_tooltip(label, LABEL_TOOLTIPS["size"])
        row.append(label)

        # Size preset dropdown
        presets = list(SIZE_PRESETS.keys())
        self._size_dropdown = Gtk.DropDown.new_from_strings(presets)
        self._size_dropdown.set_hexpand(True)
        self._size_dropdown.connect("notify::selected", self._on_size_preset_changed)
        row.append(self._size_dropdown)

        # W label and spin
        width_label = Gtk.Label(label="W")
        width_label.add_css_class("caption")
        width_label.set_margin_start(4)
        row.append(width_label)

        self._width_spin = Gtk.SpinButton.new_with_range(MIN_SIZE, MAX_SIZE, SIZE_STEP)
        self._width_spin.set_value(DEFAULT_WIDTH)
        self._width_spin.set_width_chars(5)
        self._width_spin.connect("value-changed", self._on_custom_size_changed)
        row.append(self._width_spin)

        # H label and spin
        height_label = Gtk.Label(label="H")
        height_label.add_css_class("caption")
        height_label.set_margin_start(4)
        row.append(height_label)

        self._height_spin = Gtk.SpinButton.new_with_range(MIN_SIZE, MAX_SIZE, SIZE_STEP)
        self._height_spin.set_value(DEFAULT_HEIGHT)
        self._height_spin.set_width_chars(5)
        self._height_spin.connect("value-changed", self._on_custom_size_changed)
        row.append(self._height_spin)

        return row

    def _create_sampler_section(self) -> Gtk.Widget:
        """Create sampler and scheduler selection on the same row."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Sampler
        sampler_label = self._create_label("Sampler")
        add_hover_tooltip(sampler_label, LABEL_TOOLTIPS["sampler"])
        row.append(sampler_label)

        sampler_names = list(SAMPLERS.keys())
        self._sampler_dropdown = Gtk.DropDown.new_from_strings(sampler_names)
        self._sampler_dropdown.set_hexpand(True)
        row.append(self._sampler_dropdown)

        # Scheduler
        scheduler_label = Gtk.Label(label="Sched")
        scheduler_label.add_css_class("caption")
        scheduler_label.set_margin_start(8)
        add_hover_tooltip(scheduler_label, LABEL_TOOLTIPS["scheduler"])
        row.append(scheduler_label)

        scheduler_names = SCHEDULERS
        self._scheduler_dropdown = Gtk.DropDown.new_from_strings(scheduler_names)
        self._scheduler_dropdown.set_hexpand(True)
        row.append(self._scheduler_dropdown)

        return row

    def _create_steps_cfg_section(self) -> Gtk.Widget:
        """Create steps and CFG scale on the same row."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Steps
        steps_label = self._create_label("Steps")
        add_hover_tooltip(steps_label, LABEL_TOOLTIPS["steps"])
        row.append(steps_label)

        self._steps_spin = Gtk.SpinButton.new_with_range(MIN_STEPS, MAX_STEPS, 1)
        self._steps_spin.set_value(DEFAULT_STEPS)
        self._steps_spin.set_width_chars(4)
        row.append(self._steps_spin)

        # CFG Scale
        cfg_label = Gtk.Label(label="CFG")
        cfg_label.add_css_class("caption")
        cfg_label.set_margin_start(8)
        add_hover_tooltip(cfg_label, LABEL_TOOLTIPS["cfg_scale"])
        row.append(cfg_label)

        self._cfg_spin = Gtk.SpinButton.new_with_range(
            MIN_CFG_SCALE, MAX_CFG_SCALE, 0.5
        )
        self._cfg_spin.set_digits(1)
        self._cfg_spin.set_value(DEFAULT_CFG_SCALE)
        self._cfg_spin.set_width_chars(4)
        row.append(self._cfg_spin)

        return row

    def _create_strength_section(self) -> Gtk.Widget:
        """Create strength input section (for img2img)."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        label = self._create_label("Strength")
        add_hover_tooltip(label, LABEL_TOOLTIPS["strength"])
        row.append(label)

        self._strength_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
        self._strength_spin.set_digits(2)
        self._strength_spin.set_value(0.75)
        self._strength_spin.set_hexpand(True)
        self._strength_spin.set_tooltip_text("Lower values keep more of the original image")
        row.append(self._strength_spin)

        return row

    def _create_refiner_strength_section(self) -> Gtk.Widget:
        """Create refiner strength input section (shown only in refiner mode)."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        label = self._create_label("Ref Str")
        label.set_tooltip_text("Refiner Strength: Lower values preserve more of the original region")
        label.add_css_class("refiner-label")
        row.append(label)

        self._refiner_strength_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
        self._refiner_strength_spin.set_digits(2)
        self._refiner_strength_spin.set_value(0.65)  # Default slightly lower than img2img
        self._refiner_strength_spin.set_hexpand(True)
        self._refiner_strength_spin.set_tooltip_text("Lower values keep more of the original region structure")
        row.append(self._refiner_strength_spin)

        return row

    def _create_seed_section(self) -> Gtk.Widget:
        """Create seed input section."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        label = self._create_label("Seed")
        add_hover_tooltip(label, LABEL_TOOLTIPS["seed"])
        row.append(label)

        self._seed_spin = Gtk.SpinButton.new_with_range(-1, 2**32 - 1, 1)
        self._seed_spin.set_value(DEFAULT_SEED)
        self._seed_spin.set_hexpand(True)
        row.append(self._seed_spin)

        # Random button (compact)
        random_button = Gtk.Button()
        random_button.set_icon_name("media-playlist-shuffle-symbolic")
        random_button.set_tooltip_text("Random seed (-1)")
        random_button.add_css_class("flat")
        random_button.connect("clicked", lambda b: self._seed_spin.set_value(-1))
        row.append(random_button)

        return row

    def _load_defaults(self):
        """Load default values from config."""
        config = config_manager.config.generation

        self._width_spin.set_value(config.default_width)
        self._height_spin.set_value(config.default_height)
        self._steps_spin.set_value(config.default_steps)
        self._cfg_spin.set_value(config.default_cfg_scale)
        self._seed_spin.set_value(config.default_seed)

        # Set sampler dropdown
        sampler_names = list(SAMPLERS.keys())
        if config.default_sampler in sampler_names:
            self._sampler_dropdown.set_selected(
                sampler_names.index(config.default_sampler)
            )

        # Set scheduler dropdown
        scheduler_names = SCHEDULERS
        if config.default_scheduler in scheduler_names:
            self._scheduler_dropdown.set_selected(
                scheduler_names.index(config.default_scheduler)
            )

        # Set size preset if it matches
        self._update_size_preset_from_values()

    def _on_size_preset_changed(self, dropdown: Gtk.DropDown, param):
        """Handle size preset selection."""
        selected = dropdown.get_selected()
        if selected == Gtk.INVALID_LIST_POSITION:
            return

        preset_names = list(SIZE_PRESETS.keys())
        if 0 <= selected < len(preset_names):
            preset_name = preset_names[selected]
            width, height = SIZE_PRESETS[preset_name]

            # Block signal to avoid recursion
            self._width_spin.handler_block_by_func(self._on_custom_size_changed)
            self._height_spin.handler_block_by_func(self._on_custom_size_changed)

            self._width_spin.set_value(width)
            self._height_spin.set_value(height)

            self._width_spin.handler_unblock_by_func(self._on_custom_size_changed)
            self._height_spin.handler_unblock_by_func(self._on_custom_size_changed)

    def _on_custom_size_changed(self, spin: Gtk.SpinButton):
        """Handle custom size change."""
        self._update_size_preset_from_values()

    def _update_size_preset_from_values(self):
        """Update size preset dropdown based on current values."""
        width = int(self._width_spin.get_value())
        height = int(self._height_spin.get_value())

        preset_names = list(SIZE_PRESETS.keys())
        for i, name in enumerate(preset_names):
            if SIZE_PRESETS[name] == (width, height):
                self._size_dropdown.set_selected(i)
                return

    def get_params(self, prompt: str, negative_prompt: str) -> GenerationParams:
        """Get generation parameters from current widget state."""
        sampler_names = list(SAMPLERS.keys())
        selected_sampler = self._sampler_dropdown.get_selected()
        sampler = sampler_names[selected_sampler] if 0 <= selected_sampler < len(sampler_names) else DEFAULT_SAMPLER

        scheduler_names = SCHEDULERS
        selected_scheduler = self._scheduler_dropdown.get_selected()
        scheduler = scheduler_names[selected_scheduler] if 0 <= selected_scheduler < len(scheduler_names) else DEFAULT_SCHEDULER

        return GenerationParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(self._width_spin.get_value()),
            height=int(self._height_spin.get_value()),
            steps=int(self._steps_spin.get_value()),
            cfg_scale=self._cfg_spin.get_value(),
            seed=int(self._seed_spin.get_value()),
            sampler=sampler,
            scheduler=scheduler,
        )

    def set_seed(self, seed: int):
        """Set the seed value."""
        self._seed_spin.set_value(seed)

    def set_size(self, width: int, height: int):
        """Set the image size."""
        self._width_spin.set_value(width)
        self._height_spin.set_value(height)
        self._update_size_preset_from_values()

    def set_steps(self, steps: int):
        """Set the number of steps."""
        self._steps_spin.set_value(steps)

    def set_cfg_scale(self, cfg_scale: float):
        """Set the CFG scale."""
        self._cfg_spin.set_value(cfg_scale)

    def set_sampler(self, sampler: str):
        """Set the sampler by name."""
        sampler_names = list(SAMPLERS.keys())
        if sampler in sampler_names:
            self._sampler_dropdown.set_selected(sampler_names.index(sampler))

    def set_scheduler(self, scheduler: str):
        """Set the scheduler by name."""
        scheduler_names = SCHEDULERS
        if scheduler in scheduler_names:
            self._scheduler_dropdown.set_selected(scheduler_names.index(scheduler))

    def get_strength(self) -> float:
        """Get the img2img strength value."""
        return self._strength_spin.get_value()

    def set_strength(self, strength: float):
        """Set the img2img strength value."""
        self._strength_spin.set_value(strength)

    def get_refiner_strength(self) -> float:
        """Get the refiner strength value."""
        return self._refiner_strength_spin.get_value()

    def set_refiner_strength(self, strength: float):
        """Set the refiner strength value."""
        self._refiner_strength_spin.set_value(strength)

    def set_refiner_mode(self, enabled: bool):
        """Show or hide refiner-specific controls."""
        self._refiner_strength_row.set_visible(enabled)

    def reset_to_defaults(self):
        """Reset all parameters to defaults."""
        self._load_defaults()
        self._strength_spin.set_value(0.75)
