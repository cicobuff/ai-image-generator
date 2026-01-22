"""Thumbnail gallery widget for displaying generated images."""

from pathlib import Path
from typing import Optional, Callable, List
from enum import Enum
import io

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GdkPixbuf, GLib

from PIL import Image

from src.utils.constants import THUMBNAIL_SIZE, THUMBNAIL_COLUMNS


class SortOrder(Enum):
    """Sort order for thumbnails."""
    DATE_DESC = "date_desc"  # Newest first
    DATE_ASC = "date_asc"    # Oldest first


class ThumbnailItem(Gtk.Button):
    """Single thumbnail item in the gallery."""

    def __init__(
        self,
        path: Path,
        thumbnail_size: int = THUMBNAIL_SIZE,
        image: Optional[Image.Image] = None,
        on_click: Optional[Callable[[Path], None]] = None,
    ):
        super().__init__()
        self._path = path
        self._on_click = on_click
        self._thumbnail_size = thumbnail_size
        self._original_image = image

        self.add_css_class("thumbnail")
        self.set_has_frame(False)

        self._build_ui(image)

        self.connect("clicked", self._on_clicked)

    def _build_ui(self, image: Optional[Image.Image]):
        """Build the thumbnail UI."""
        # Create thumbnail image
        if image:
            pixbuf = self._pil_to_thumbnail(image)
        else:
            # Load from path
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    str(self._path),
                    self._thumbnail_size,
                    self._thumbnail_size,
                    True,  # Preserve aspect ratio
                )
            except Exception:
                pixbuf = None

        if pixbuf:
            picture = Gtk.Picture.new_for_pixbuf(pixbuf)
            picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            picture.set_size_request(self._thumbnail_size, self._thumbnail_size)
            self.set_child(picture)
        else:
            # Placeholder
            label = Gtk.Label(label="?")
            label.set_size_request(self._thumbnail_size, self._thumbnail_size)
            self.set_child(label)

    def _pil_to_thumbnail(self, image: Image.Image) -> Optional[GdkPixbuf.Pixbuf]:
        """Convert PIL Image to thumbnail GdkPixbuf."""
        try:
            # Create thumbnail
            thumb = image.copy()
            thumb.thumbnail((self._thumbnail_size, self._thumbnail_size))

            # Convert to bytes
            buffer = io.BytesIO()
            thumb.save(buffer, format="PNG")
            buffer.seek(0)

            # Load into GdkPixbuf
            loader = GdkPixbuf.PixbufLoader.new_with_type("png")
            loader.write(buffer.read())
            loader.close()

            return loader.get_pixbuf()
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None

    def _on_clicked(self, button):
        """Handle thumbnail click."""
        if self._on_click:
            self._on_click(self._path)

    @property
    def path(self) -> Path:
        """Get the image path."""
        return self._path

    def set_selected(self, selected: bool):
        """Set the selection state."""
        if selected:
            self.add_css_class("thumbnail-selected")
        else:
            self.remove_css_class("thumbnail-selected")

    def update_size(self, new_size: int):
        """Update the thumbnail size."""
        self._thumbnail_size = new_size
        # Rebuild the UI with new size
        self._build_ui(self._original_image)


class ThumbnailGallery(Gtk.Box):
    """Scrollable gallery of image thumbnails with toolbar and directory selector."""

    MIN_THUMBNAIL_SIZE = 64
    MAX_THUMBNAIL_SIZE = 256
    DEFAULT_THUMBNAIL_SIZE = THUMBNAIL_SIZE

    def __init__(
        self,
        on_image_selected: Optional[Callable[[Path], None]] = None,
        on_directory_changed: Optional[Callable[[Path], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._on_image_selected = on_image_selected
        self._on_directory_changed = on_directory_changed
        self._thumbnails: List[ThumbnailItem] = []
        self._image_paths: List[Path] = []  # Store paths for re-sorting
        self._selected_path: Optional[Path] = None
        self._thumbnail_size = self.DEFAULT_THUMBNAIL_SIZE
        self._sort_order = SortOrder.DATE_DESC
        self._base_directory: Optional[Path] = None  # Root output directory
        self._current_directory: Optional[Path] = None  # Currently selected subdirectory
        self._subdirectories: List[str] = []  # List of subdirectory names

        self.add_css_class("thumbnail-gallery")
        self._build_ui()

    def _build_ui(self):
        """Build the gallery UI."""
        # Header
        header = Gtk.Label(label="Gallery")
        header.add_css_class("section-header")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Directory selector row: Label | Combo | Refresh
        dir_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(dir_row)

        # Directory label
        dir_label = Gtk.Label(label="Folder")
        dir_label.set_size_request(45, -1)
        dir_label.set_halign(Gtk.Align.START)
        dir_label.add_css_class("caption")
        dir_row.append(dir_label)

        # Editable combo box for directory selection
        self._dir_combo = Gtk.ComboBoxText.new_with_entry()
        self._dir_combo.set_hexpand(True)
        self._dir_combo.set_tooltip_text("Select or type a subdirectory name")
        self._dir_combo.connect("changed", self._on_directory_combo_changed)
        dir_row.append(self._dir_combo)

        # Refresh directory list button
        refresh_btn = Gtk.Button()
        refresh_btn.set_icon_name("view-refresh-symbolic")
        refresh_btn.set_tooltip_text("Refresh directory list")
        refresh_btn.add_css_class("flat")
        refresh_btn.connect("clicked", self._on_refresh_directories)
        dir_row.append(refresh_btn)

        # Controls row: Sort | Size slider
        controls_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(controls_row)

        # Sort label
        sort_label = Gtk.Label(label="Sort")
        sort_label.set_size_request(45, -1)
        sort_label.set_halign(Gtk.Align.START)
        sort_label.add_css_class("caption")
        controls_row.append(sort_label)

        # Sort toggle button
        self._sort_button = Gtk.Button()
        self._sort_button.set_icon_name("view-sort-descending-symbolic")
        self._sort_button.set_tooltip_text("Sort: Newest first (click to change)")
        self._sort_button.add_css_class("flat")
        self._sort_button.connect("clicked", self._on_sort_clicked)
        controls_row.append(self._sort_button)

        # Size label
        size_label = Gtk.Label(label="Size")
        size_label.set_margin_start(8)
        size_label.add_css_class("caption")
        controls_row.append(size_label)

        # Thumbnail size slider
        self._size_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL,
            self.MIN_THUMBNAIL_SIZE,
            self.MAX_THUMBNAIL_SIZE,
            16
        )
        self._size_scale.set_value(self._thumbnail_size)
        self._size_scale.set_draw_value(False)
        self._size_scale.set_hexpand(True)
        self._size_scale.set_size_request(60, -1)
        self._size_scale.set_tooltip_text(f"Thumbnail size: {self._thumbnail_size}px")
        self._size_scale.connect("value-changed", self._on_size_changed)
        controls_row.append(self._size_scale)

        # Scrolled window
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_margin_top(4)
        self.append(scrolled)

        # Flow box for thumbnails
        self._flowbox = Gtk.FlowBox()
        self._flowbox.set_valign(Gtk.Align.START)
        self._flowbox.set_max_children_per_line(10)
        self._flowbox.set_min_children_per_line(1)
        self._flowbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self._flowbox.set_homogeneous(True)
        self._flowbox.set_row_spacing(4)
        self._flowbox.set_column_spacing(4)
        scrolled.set_child(self._flowbox)

    def _scan_subdirectories(self):
        """Scan for subdirectories in the base directory."""
        self._subdirectories = []
        if self._base_directory is None or not self._base_directory.exists():
            return

        # Get all subdirectories
        for item in self._base_directory.iterdir():
            if item.is_dir():
                self._subdirectories.append(item.name)

        # Sort alphabetically
        self._subdirectories.sort()

    def _update_directory_combo(self):
        """Update the directory combo box with current subdirectories."""
        # Block signal while updating
        self._dir_combo.handler_block_by_func(self._on_directory_combo_changed)

        # Clear existing items
        self._dir_combo.remove_all()

        # Add root directory option (empty string means root)
        self._dir_combo.append_text("(root)")

        # Add subdirectories
        for subdir in self._subdirectories:
            self._dir_combo.append_text(subdir)

        # Set active based on current directory
        if self._current_directory == self._base_directory:
            self._dir_combo.set_active(0)
        else:
            # Find the subdirectory in the list
            rel_path = self._get_relative_subdir()
            if rel_path in self._subdirectories:
                idx = self._subdirectories.index(rel_path) + 1  # +1 for (root)
                self._dir_combo.set_active(idx)
            else:
                # Custom typed directory - set text in entry
                entry = self._dir_combo.get_child()
                if entry and rel_path:
                    entry.set_text(rel_path)

        # Unblock signal
        self._dir_combo.handler_unblock_by_func(self._on_directory_combo_changed)

    def _get_relative_subdir(self) -> str:
        """Get the relative subdirectory name from current directory."""
        if self._current_directory is None or self._base_directory is None:
            return ""
        if self._current_directory == self._base_directory:
            return ""
        try:
            rel = self._current_directory.relative_to(self._base_directory)
            return str(rel)
        except ValueError:
            return ""

    def _on_directory_combo_changed(self, combo):
        """Handle directory selection change."""
        if self._base_directory is None:
            return

        # Get the selected/typed text
        active_idx = combo.get_active()
        if active_idx == 0:
            # Root directory selected
            new_dir = self._base_directory
        elif active_idx > 0:
            # Existing subdirectory selected
            subdir_name = self._subdirectories[active_idx - 1]
            new_dir = self._base_directory / subdir_name
        else:
            # Custom text typed
            entry = combo.get_child()
            if entry:
                text = entry.get_text().strip()
                if text and text != "(root)":
                    new_dir = self._base_directory / text
                else:
                    new_dir = self._base_directory
            else:
                new_dir = self._base_directory

        # Update current directory
        if new_dir != self._current_directory:
            self._current_directory = new_dir

            # Notify callback
            if self._on_directory_changed:
                self._on_directory_changed(new_dir)

            # Refresh thumbnails if directory exists
            if new_dir.exists():
                self._load_thumbnails_from_current_directory()
            else:
                # Directory doesn't exist - clear thumbnails
                self._clear_thumbnails()

    def _on_refresh_directories(self, button):
        """Handle refresh directories button click."""
        self._scan_subdirectories()
        self._update_directory_combo()

    def _on_sort_clicked(self, button):
        """Handle sort button click."""
        # Toggle sort order
        if self._sort_order == SortOrder.DATE_DESC:
            self._sort_order = SortOrder.DATE_ASC
            self._sort_button.set_icon_name("view-sort-ascending-symbolic")
            self._sort_button.set_tooltip_text("Sort: Oldest first (click to change)")
        else:
            self._sort_order = SortOrder.DATE_DESC
            self._sort_button.set_icon_name("view-sort-descending-symbolic")
            self._sort_button.set_tooltip_text("Sort: Newest first (click to change)")

        # Re-sort and refresh
        self._refresh_thumbnails()

    def _on_size_changed(self, scale):
        """Handle thumbnail size change."""
        new_size = int(scale.get_value())
        if new_size == self._thumbnail_size:
            return

        self._thumbnail_size = new_size
        self._size_scale.set_tooltip_text(f"Thumbnail size: {new_size}px")

        # Refresh thumbnails with new size
        self._refresh_thumbnails()

    def _clear_thumbnails(self):
        """Clear all thumbnail widgets without clearing paths."""
        for thumb in self._thumbnails:
            self._flowbox.remove(thumb)
        self._thumbnails.clear()
        self._image_paths.clear()
        self._selected_path = None

    def _refresh_thumbnails(self):
        """Refresh all thumbnails with current settings."""
        if not self._image_paths:
            return

        # Store selected path
        selected = self._selected_path

        # Clear current thumbnails
        for thumb in self._thumbnails:
            self._flowbox.remove(thumb)
        self._thumbnails.clear()

        # Sort paths
        sorted_paths = self._sort_paths(self._image_paths)

        # Recreate thumbnails
        for path in sorted_paths:
            thumbnail = ThumbnailItem(
                path=path,
                thumbnail_size=self._thumbnail_size,
                on_click=self._on_thumbnail_clicked,
            )
            self._thumbnails.append(thumbnail)
            self._flowbox.append(thumbnail)

        # Restore selection
        if selected:
            for thumb in self._thumbnails:
                thumb.set_selected(thumb.path == selected)

    def _sort_paths(self, paths: List[Path]) -> List[Path]:
        """Sort paths according to current sort order."""
        reverse = self._sort_order == SortOrder.DATE_DESC
        return sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=reverse)

    def _load_thumbnails_from_current_directory(self):
        """Load thumbnails from the current directory (not subdirectories)."""
        self._clear_thumbnails()

        if self._current_directory is None or not self._current_directory.exists():
            return

        # Get image files (only in current directory, not subdirectories)
        image_files = []
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            # Use iterdir to avoid recursion into subdirectories
            for item in self._current_directory.iterdir():
                if item.is_file() and item.suffix.lower() == ext:
                    image_files.append(item)

        # Store paths
        self._image_paths = image_files

        # Sort according to current order
        sorted_files = self._sort_paths(image_files)

        # Add thumbnails
        for path in sorted_files:
            thumbnail = ThumbnailItem(
                path=path,
                thumbnail_size=self._thumbnail_size,
                on_click=self._on_thumbnail_clicked,
            )
            self._thumbnails.append(thumbnail)
            self._flowbox.append(thumbnail)

    def add_image(self, path: Path, image: Optional[Image.Image] = None):
        """Add a new image to the gallery."""
        # Only add if the image is in the current directory
        if self._current_directory and path.parent == self._current_directory:
            # Add to paths list
            if path not in self._image_paths:
                self._image_paths.append(path)

            thumbnail = ThumbnailItem(
                path=path,
                thumbnail_size=self._thumbnail_size,
                image=image,
                on_click=self._on_thumbnail_clicked,
            )

            # Insert based on sort order
            if self._sort_order == SortOrder.DATE_DESC:
                # Newest first - prepend
                self._thumbnails.insert(0, thumbnail)
                self._flowbox.prepend(thumbnail)
            else:
                # Oldest first - append
                self._thumbnails.append(thumbnail)
                self._flowbox.append(thumbnail)

    def _on_thumbnail_clicked(self, path: Path):
        """Handle thumbnail click."""
        # Update selection state
        self._selected_path = path
        for thumb in self._thumbnails:
            thumb.set_selected(thumb.path == path)

        # Notify callback
        if self._on_image_selected:
            self._on_image_selected(path)

    def clear(self):
        """Clear all thumbnails."""
        self._clear_thumbnails()

    def load_from_directory(self, directory: Path):
        """Set the base directory and load thumbnails."""
        self._base_directory = directory
        self._current_directory = directory

        # Ensure directory exists
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

        # Scan for subdirectories
        self._scan_subdirectories()
        self._update_directory_combo()

        # Load thumbnails from root
        self._load_thumbnails_from_current_directory()

    def get_selected_path(self) -> Optional[Path]:
        """Get the currently selected image path."""
        return self._selected_path

    def select_path(self, path: Path):
        """Select a specific image by path."""
        self._on_thumbnail_clicked(path)

    def get_output_directory(self) -> Path:
        """Get the current output directory for saving new images."""
        if self._current_directory is not None:
            return self._current_directory
        if self._base_directory is not None:
            return self._base_directory
        # Fallback
        from src.core.config import config_manager
        return config_manager.config.get_output_path()

    def set_subdirectory(self, subdir_name: str):
        """Set the current subdirectory by name."""
        if self._base_directory is None:
            return

        if not subdir_name or subdir_name == "(root)":
            new_dir = self._base_directory
        else:
            new_dir = self._base_directory / subdir_name

        self._current_directory = new_dir

        # Update combo box
        entry = self._dir_combo.get_child()
        if entry:
            if subdir_name and subdir_name != "(root)":
                entry.set_text(subdir_name)
            else:
                self._dir_combo.set_active(0)

        # Notify callback
        if self._on_directory_changed:
            self._on_directory_changed(new_dir)

        # Refresh thumbnails if directory exists
        if new_dir.exists():
            self._load_thumbnails_from_current_directory()
        else:
            self._clear_thumbnails()

    @property
    def thumbnail_size(self) -> int:
        """Get current thumbnail size."""
        return self._thumbnail_size

    @thumbnail_size.setter
    def thumbnail_size(self, size: int):
        """Set thumbnail size."""
        size = max(self.MIN_THUMBNAIL_SIZE, min(size, self.MAX_THUMBNAIL_SIZE))
        self._size_scale.set_value(size)

    @property
    def sort_order(self) -> SortOrder:
        """Get current sort order."""
        return self._sort_order

    @property
    def current_directory(self) -> Optional[Path]:
        """Get the current directory being displayed."""
        return self._current_directory

    @property
    def base_directory(self) -> Optional[Path]:
        """Get the base output directory."""
        return self._base_directory
