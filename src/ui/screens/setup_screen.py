"""Setup screen for first-launch configuration."""

from pathlib import Path
from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gio

from src.core.config import config_manager, AppConfig, DirectoriesConfig


class SetupScreen(Gtk.Box):
    """First-launch setup screen for configuring directories."""

    def __init__(self, on_complete: Callable[[], None]):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._on_complete = on_complete
        self.add_css_class("setup-screen")
        self._build_ui()

    def _build_ui(self):
        """Build the setup screen UI."""
        # Scrollable container
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        self.append(scrolled)

        # Main content box
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        content.set_margin_top(24)
        content.set_margin_bottom(24)
        content.set_margin_start(48)
        content.set_margin_end(48)
        content.set_halign(Gtk.Align.CENTER)
        content.set_valign(Gtk.Align.START)
        scrolled.set_child(content)

        # Title
        title = Gtk.Label(label="AI Image Generator Setup")
        title.add_css_class("setup-title")
        title.add_css_class("title-1")
        content.append(title)

        # Subtitle
        subtitle = Gtk.Label(
            label="Configure your directories to get started."
        )
        subtitle.add_css_class("dim-label")
        content.append(subtitle)

        # Directories section
        content.append(self._create_directories_section())

        # Save button
        save_button = Gtk.Button(label="Save and Continue")
        save_button.add_css_class("suggested-action")
        save_button.add_css_class("pill")
        save_button.set_halign(Gtk.Align.CENTER)
        save_button.set_margin_top(24)
        save_button.connect("clicked", self._on_save_clicked)
        content.append(save_button)

    def _create_directories_section(self) -> Gtk.Widget:
        """Create the directories configuration section."""
        frame = Gtk.Frame()
        frame.set_label("Directories")

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        frame.set_child(box)

        # Get current config values
        config = config_manager.config

        # Models directory
        models_row = self._create_directory_row(
            "Models Directory:",
            config.directories.models,
            "models_entry",
        )
        box.append(models_row)

        # Output directory
        output_row = self._create_directory_row(
            "Output Directory:",
            config.directories.output,
            "output_entry",
        )
        box.append(output_row)

        return frame

    def _create_directory_row(
        self, label_text: str, default_value: str, entry_name: str
    ) -> Gtk.Widget:
        """Create a directory input row with browse button."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_size_request(150, -1)
        row.append(label)

        entry = Gtk.Entry()
        entry.set_text(default_value)
        entry.set_hexpand(True)
        entry.set_name(entry_name)
        row.append(entry)

        # Store reference to entry
        setattr(self, f"_{entry_name}", entry)

        browse_button = Gtk.Button(label="Browse...")
        browse_button.connect(
            "clicked", lambda b: self._on_browse_clicked(entry)
        )
        row.append(browse_button)

        return row

    def _on_browse_clicked(self, entry: Gtk.Entry):
        """Handle browse button click."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Directory")

        # Set initial folder if entry has a value
        current = entry.get_text()
        if current:
            path = Path(current)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                dialog.set_initial_folder(Gio.File.new_for_path(str(path)))

        dialog.select_folder(
            self.get_root(),
            None,
            lambda d, r: self._on_folder_selected(d, r, entry),
        )

    def _on_folder_selected(
        self, dialog: Gtk.FileDialog, result, entry: Gtk.Entry
    ):
        """Handle folder selection result."""
        try:
            folder = dialog.select_folder_finish(result)
            if folder:
                entry.set_text(folder.get_path())
        except Exception:
            pass  # User cancelled

    def _on_save_clicked(self, button: Gtk.Button):
        """Handle save button click."""
        # Gather directory values
        models_dir = self._models_entry.get_text().strip()
        output_dir = self._output_entry.get_text().strip()

        if not models_dir:
            models_dir = "./models"
        if not output_dir:
            output_dir = "./output"

        # Create and save config
        config = AppConfig(
            directories=DirectoriesConfig(
                models=models_dir,
                output=output_dir,
            ),
        )

        config_manager.save(config)

        # Call completion callback
        self._on_complete()


