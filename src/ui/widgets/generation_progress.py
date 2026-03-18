"""Generation progress widget showing batch and step progress."""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.ui.widgets.info_helper import SectionHeader, SECTION_INFO


class GenerationProgressWidget(Gtk.Box):
    """Widget showing generation progress with batch and step progress bars."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)

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

        # Step progress bar
        step_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(step_box)

        step_label = Gtk.Label(label="Step")
        step_label.set_size_request(70, -1)
        step_label.set_halign(Gtk.Align.START)
        step_label.add_css_class("caption")
        step_box.append(step_label)

        self._step_progress = Gtk.ProgressBar()
        self._step_progress.set_hexpand(True)
        self._step_progress.set_valign(Gtk.Align.CENTER)
        self._step_progress.add_css_class("generation-progress-bar")
        step_box.append(self._step_progress)

        self._step_label = Gtk.Label(label="")
        self._step_label.set_size_request(50, -1)
        self._step_label.set_halign(Gtk.Align.END)
        self._step_label.add_css_class("caption")
        self._step_label.add_css_class("monospace")
        step_box.append(self._step_label)

        # Status text
        self._status_label = Gtk.Label(label="Ready")
        self._status_label.set_halign(Gtk.Align.START)
        self._status_label.add_css_class("caption")
        self._status_label.set_ellipsize(True)
        self._status_label.set_max_width_chars(40)
        self.append(self._status_label)

        # Batch summary box (hidden by default)
        self._summary_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._summary_box.set_visible(False)
        self._summary_box.add_css_class("batch-summary")
        self._summary_box.set_margin_top(8)
        self.append(self._summary_box)

        # Summary labels
        self._summary_header = Gtk.Label(label="Batch Summary")
        self._summary_header.set_halign(Gtk.Align.START)
        self._summary_header.add_css_class("caption")
        self._summary_header.add_css_class("dim-label")
        self._summary_box.append(self._summary_header)

        self._summary_images_label = Gtk.Label()
        self._summary_images_label.set_halign(Gtk.Align.START)
        self._summary_images_label.add_css_class("caption")
        self._summary_box.append(self._summary_images_label)

        self._summary_time_label = Gtk.Label()
        self._summary_time_label.set_halign(Gtk.Align.START)
        self._summary_time_label.add_css_class("caption")
        self._summary_box.append(self._summary_time_label)

        self._summary_avg_label = Gtk.Label()
        self._summary_avg_label.set_halign(Gtk.Align.START)
        self._summary_avg_label.add_css_class("caption")
        self._summary_box.append(self._summary_avg_label)

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

        # Hide batch summary
        self._summary_box.set_visible(False)

    def show_batch_summary(self, images_generated: int, total_seconds: float, was_cancelled: bool = False):
        """Show batch generation summary.

        Args:
            images_generated: Number of images successfully generated
            total_seconds: Total time taken in seconds
            was_cancelled: Whether the batch was cancelled
        """
        # Format total time
        if total_seconds >= 60:
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{total_seconds:.1f}s"

        # Calculate average time per image
        if images_generated > 0:
            avg_seconds = total_seconds / images_generated
            if avg_seconds >= 60:
                avg_minutes = int(avg_seconds // 60)
                avg_secs = avg_seconds % 60
                avg_str = f"{avg_minutes}m {avg_secs:.1f}s"
            else:
                avg_str = f"{avg_seconds:.1f}s"
        else:
            avg_str = "N/A"

        # Update labels
        status = "Cancelled" if was_cancelled else "Complete"
        self._summary_header.set_text(f"Batch {status}")
        self._summary_images_label.set_text(f"Images: {images_generated}")
        self._summary_time_label.set_text(f"Total time: {time_str}")
        self._summary_avg_label.set_text(f"Avg per image: {avg_str}")

        # Show the summary box
        self._summary_box.set_visible(True)

    def set_generating(self, is_generating: bool):
        """Set whether currently generating (affects visual state)."""
        if is_generating:
            self._batch_progress.add_css_class("generating")
            self._step_progress.add_css_class("generating")
        else:
            self._batch_progress.remove_css_class("generating")
            self._step_progress.remove_css_class("generating")
