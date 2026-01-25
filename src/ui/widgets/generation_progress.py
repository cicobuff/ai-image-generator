"""Generation progress widget showing batch and step progress."""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.ui.widgets.info_helper import SectionHeader, SECTION_INFO


class GenerationProgressWidget(Gtk.Box):
    """Widget showing generation progress with batch and step progress bars.

    Supports both single-image generation (one Step bar) and batch generation
    with per-GPU step progress bars (Step (GPU0), Step (GPU1), etc.).
    """

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        # Track GPU progress bars for batch mode
        self._gpu_progress_rows: dict[int, tuple[Gtk.Box, Gtk.ProgressBar, Gtk.Label]] = {}

        self._build_ui()

    def _build_ui(self):
        """Build the progress widget UI."""
        # Section header
        header = SectionHeader("Generation Progress", SECTION_INFO.get("generation_progress", ""))
        self.append(header)

        # Batch progress (total generations)
        batch_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(batch_box)

        batch_label = Gtk.Label(label="Batch")
        batch_label.set_size_request(70, -1)
        batch_label.set_halign(Gtk.Align.START)
        batch_label.add_css_class("caption")
        batch_box.append(batch_label)

        self._batch_progress = Gtk.ProgressBar()
        self._batch_progress.set_hexpand(True)
        self._batch_progress.set_valign(Gtk.Align.CENTER)
        self._batch_progress.add_css_class("generation-progress-bar")
        batch_box.append(self._batch_progress)

        self._batch_label = Gtk.Label(label="")
        self._batch_label.set_size_request(50, -1)
        self._batch_label.set_halign(Gtk.Align.END)
        self._batch_label.add_css_class("caption")
        self._batch_label.add_css_class("monospace")
        batch_box.append(self._batch_label)

        # Step progress container - will hold either single Step bar or multiple GPU bars
        self._step_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.append(self._step_container)

        # Default single step progress bar (for single-image generation)
        self._single_step_box = self._create_step_row("Step")
        self._step_container.append(self._single_step_box)

        # Store references to single step widgets
        self._step_progress = self._single_step_box.get_first_child().get_next_sibling()
        self._step_label = self._step_progress.get_next_sibling()

        # Status text
        self._status_label = Gtk.Label(label="Ready")
        self._status_label.set_halign(Gtk.Align.START)
        self._status_label.add_css_class("caption")
        self._status_label.set_ellipsize(True)
        self._status_label.set_max_width_chars(40)
        self.append(self._status_label)

    def _create_step_row(self, label_text: str) -> Gtk.Box:
        """Create a step progress row with label, progress bar, and value label."""
        step_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        step_label = Gtk.Label(label=label_text)
        step_label.set_size_request(70, -1)
        step_label.set_halign(Gtk.Align.START)
        step_label.add_css_class("caption")
        step_box.append(step_label)

        step_progress = Gtk.ProgressBar()
        step_progress.set_hexpand(True)
        step_progress.set_valign(Gtk.Align.CENTER)
        step_progress.add_css_class("generation-progress-bar")
        step_box.append(step_progress)

        value_label = Gtk.Label(label="")
        value_label.set_size_request(50, -1)
        value_label.set_halign(Gtk.Align.END)
        value_label.add_css_class("caption")
        value_label.add_css_class("monospace")
        step_box.append(value_label)

        return step_box

    def setup_gpu_progress_bars(self, gpu_indices: list[int]):
        """Set up per-GPU step progress bars for batch generation.

        Args:
            gpu_indices: List of GPU indices to create progress bars for
        """
        # Remove single step bar
        self._step_container.remove(self._single_step_box)

        # Clear any existing GPU progress bars
        for gpu_idx in list(self._gpu_progress_rows.keys()):
            row, _, _ = self._gpu_progress_rows[gpu_idx]
            self._step_container.remove(row)
        self._gpu_progress_rows.clear()

        # Create progress bars for each GPU
        for gpu_idx in gpu_indices:
            label_text = f"Step (GPU{gpu_idx})"
            row = self._create_step_row(label_text)
            self._step_container.append(row)

            # Get references to the progress bar and label
            progress_bar = row.get_first_child().get_next_sibling()
            value_label = progress_bar.get_next_sibling()
            progress_bar.add_css_class("generating")

            self._gpu_progress_rows[gpu_idx] = (row, progress_bar, value_label)

    def clear_gpu_progress_bars(self):
        """Remove per-GPU progress bars and restore single Step bar."""
        # Remove GPU progress bars
        for gpu_idx in list(self._gpu_progress_rows.keys()):
            row, _, _ = self._gpu_progress_rows[gpu_idx]
            self._step_container.remove(row)
        self._gpu_progress_rows.clear()

        # Restore single step bar only if not already in container
        if self._single_step_box.get_parent() is None:
            self._step_container.append(self._single_step_box)

    def set_gpu_step_progress(self, gpu_idx: int, current: int, total: int):
        """Set step progress for a specific GPU.

        Args:
            gpu_idx: GPU index
            current: Current step number
            total: Total steps
        """
        if gpu_idx not in self._gpu_progress_rows:
            return

        _, progress_bar, value_label = self._gpu_progress_rows[gpu_idx]

        if total > 0:
            fraction = current / total
            progress_bar.set_fraction(fraction)
            value_label.set_text(f"{current}/{total}")
        else:
            progress_bar.set_fraction(0)
            value_label.set_text("")

    def reset_gpu_step_progress(self, gpu_idx: int):
        """Reset step progress for a specific GPU (when it finishes one image)."""
        if gpu_idx not in self._gpu_progress_rows:
            return

        _, progress_bar, value_label = self._gpu_progress_rows[gpu_idx]
        progress_bar.set_fraction(0)
        value_label.set_text("")

    def set_batch_progress(self, current: int, total: int):
        """Set the batch (total generations) progress.

        Args:
            current: Current generation number (1-indexed)
            total: Total number of generations
        """
        if total > 0:
            fraction = current / total
            self._batch_progress.set_fraction(fraction)
            self._batch_label.set_text(f"{current}/{total}")
        else:
            self._batch_progress.set_fraction(0)
            self._batch_label.set_text("")

    def set_step_progress(self, current: int, total: int):
        """Set the step progress for current generation (single-image mode).

        Args:
            current: Current step number
            total: Total steps
        """
        if total > 0:
            fraction = current / total
            self._step_progress.set_fraction(fraction)
            self._step_label.set_text(f"{current}/{total}")
        else:
            self._step_progress.set_fraction(0)
            self._step_label.set_text("")

    def set_step_fraction(self, fraction: float):
        """Set step progress as a fraction (0.0 to 1.0)."""
        self._step_progress.set_fraction(fraction)
        percent = int(fraction * 100)
        self._step_label.set_text(f"{percent}%")

    def set_status(self, message: str):
        """Set the status message."""
        self._status_label.set_text(message)

    def reset(self):
        """Reset all progress indicators."""
        self._batch_progress.set_fraction(0)
        self._batch_label.set_text("")
        self._step_progress.set_fraction(0)
        self._step_label.set_text("")
        self._status_label.set_text("Ready")

        # Clear GPU progress bars and restore single step bar
        self.clear_gpu_progress_bars()

    def set_generating(self, is_generating: bool):
        """Set whether currently generating (affects visual state)."""
        if is_generating:
            self._batch_progress.add_css_class("generating")
            self._step_progress.add_css_class("generating")
        else:
            self._batch_progress.remove_css_class("generating")
            self._step_progress.remove_css_class("generating")
