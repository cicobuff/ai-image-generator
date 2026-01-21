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


class GenerationParamsWidget(Gtk.Box):
    """Widget for configuring image generation parameters."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        """Build the parameters UI."""
        # Header
        header = Gtk.Label(label="Generation Parameters")
        header.add_css_class("section-header")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Size presets
        self.append(self._create_size_section())

        # Sampler
        self.append(self._create_sampler_section())

        # Steps
        self.append(self._create_steps_section())

        # CFG Scale
        self.append(self._create_cfg_section())

        # Seed
        self.append(self._create_seed_section())

    def _create_size_section(self) -> Gtk.Widget:
        """Create size selection section."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        # Size preset dropdown
        preset_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        preset_label = Gtk.Label(label="Size:")
        preset_label.set_size_request(80, -1)
        preset_label.set_halign(Gtk.Align.START)
        preset_row.append(preset_label)

        presets = list(SIZE_PRESETS.keys())
        self._size_dropdown = Gtk.DropDown.new_from_strings(presets)
        self._size_dropdown.set_hexpand(True)
        self._size_dropdown.connect("notify::selected", self._on_size_preset_changed)
        preset_row.append(self._size_dropdown)
        box.append(preset_row)

        # Custom width/height
        custom_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        width_label = Gtk.Label(label="W:")
        custom_row.append(width_label)

        self._width_spin = Gtk.SpinButton.new_with_range(MIN_SIZE, MAX_SIZE, SIZE_STEP)
        self._width_spin.set_value(DEFAULT_WIDTH)
        self._width_spin.connect("value-changed", self._on_custom_size_changed)
        custom_row.append(self._width_spin)

        height_label = Gtk.Label(label="H:")
        height_label.set_margin_start(8)
        custom_row.append(height_label)

        self._height_spin = Gtk.SpinButton.new_with_range(MIN_SIZE, MAX_SIZE, SIZE_STEP)
        self._height_spin.set_value(DEFAULT_HEIGHT)
        self._height_spin.connect("value-changed", self._on_custom_size_changed)
        custom_row.append(self._height_spin)

        box.append(custom_row)

        return box

    def _create_sampler_section(self) -> Gtk.Widget:
        """Create sampler selection section."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.add_css_class("param-row")

        label = Gtk.Label(label="Sampler:")
        label.set_size_request(80, -1)
        label.set_halign(Gtk.Align.START)
        row.append(label)

        sampler_names = list(SAMPLERS.keys())
        self._sampler_dropdown = Gtk.DropDown.new_from_strings(sampler_names)
        self._sampler_dropdown.set_hexpand(True)
        row.append(self._sampler_dropdown)

        return row

    def _create_steps_section(self) -> Gtk.Widget:
        """Create steps input section."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.add_css_class("param-row")

        label = Gtk.Label(label="Steps:")
        label.set_size_request(80, -1)
        label.set_halign(Gtk.Align.START)
        row.append(label)

        self._steps_spin = Gtk.SpinButton.new_with_range(MIN_STEPS, MAX_STEPS, 1)
        self._steps_spin.set_value(DEFAULT_STEPS)
        self._steps_spin.set_hexpand(True)
        row.append(self._steps_spin)

        return row

    def _create_cfg_section(self) -> Gtk.Widget:
        """Create CFG scale input section."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.add_css_class("param-row")

        label = Gtk.Label(label="CFG Scale:")
        label.set_size_request(80, -1)
        label.set_halign(Gtk.Align.START)
        row.append(label)

        self._cfg_spin = Gtk.SpinButton.new_with_range(
            MIN_CFG_SCALE, MAX_CFG_SCALE, 0.5
        )
        self._cfg_spin.set_digits(1)
        self._cfg_spin.set_value(DEFAULT_CFG_SCALE)
        self._cfg_spin.set_hexpand(True)
        row.append(self._cfg_spin)

        return row

    def _create_seed_section(self) -> Gtk.Widget:
        """Create seed input section."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.add_css_class("param-row")

        label = Gtk.Label(label="Seed:")
        label.set_size_request(80, -1)
        label.set_halign(Gtk.Align.START)
        row.append(label)

        self._seed_spin = Gtk.SpinButton.new_with_range(-1, 2**32 - 1, 1)
        self._seed_spin.set_value(DEFAULT_SEED)
        self._seed_spin.set_hexpand(True)
        row.append(self._seed_spin)

        # Random button
        random_button = Gtk.Button(label="Random")
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

        return GenerationParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(self._width_spin.get_value()),
            height=int(self._height_spin.get_value()),
            steps=int(self._steps_spin.get_value()),
            cfg_scale=self._cfg_spin.get_value(),
            seed=int(self._seed_spin.get_value()),
            sampler=sampler,
        )

    def set_seed(self, seed: int):
        """Set the seed value."""
        self._seed_spin.set_value(seed)
