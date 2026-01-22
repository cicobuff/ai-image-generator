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
    """Scrollable gallery of image thumbnails with toolbar."""

    MIN_THUMBNAIL_SIZE = 64
    MAX_THUMBNAIL_SIZE = 256
    DEFAULT_THUMBNAIL_SIZE = THUMBNAIL_SIZE

    def __init__(
        self,
        on_image_selected: Optional[Callable[[Path], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._on_image_selected = on_image_selected
        self._thumbnails: List[ThumbnailItem] = []
        self._image_paths: List[Path] = []  # Store paths for re-sorting
        self._selected_path: Optional[Path] = None
        self._thumbnail_size = self.DEFAULT_THUMBNAIL_SIZE
        self._sort_order = SortOrder.DATE_DESC
        self._current_directory: Optional[Path] = None

        self.add_css_class("thumbnail-gallery")
        self._build_ui()

    def _build_ui(self):
        """Build the gallery UI."""
        # Header
        header = Gtk.Label(label="Generated Images")
        header.add_css_class("section-header")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Mini toolbar
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        toolbar.set_margin_top(4)
        toolbar.set_margin_bottom(4)
        self.append(toolbar)

        # Sort toggle button
        self._sort_button = Gtk.Button()
        self._sort_button.set_icon_name("view-sort-descending-symbolic")
        self._sort_button.set_tooltip_text("Sort: Newest first (click to change)")
        self._sort_button.add_css_class("flat")
        self._sort_button.connect("clicked", self._on_sort_clicked)
        toolbar.append(self._sort_button)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        separator.set_margin_start(4)
        separator.set_margin_end(4)
        toolbar.append(separator)

        # Size icon (small)
        size_icon_small = Gtk.Image.new_from_icon_name("view-grid-symbolic")
        size_icon_small.set_pixel_size(12)
        toolbar.append(size_icon_small)

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
        toolbar.append(self._size_scale)

        # Size icon (large)
        size_icon_large = Gtk.Image.new_from_icon_name("view-grid-symbolic")
        size_icon_large.set_pixel_size(20)
        toolbar.append(size_icon_large)

        # Scrolled window
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
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

    def add_image(self, path: Path, image: Optional[Image.Image] = None):
        """Add a new image to the gallery."""
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
        for thumb in self._thumbnails:
            self._flowbox.remove(thumb)
        self._thumbnails.clear()
        self._image_paths.clear()
        self._selected_path = None

    def load_from_directory(self, directory: Path):
        """Load thumbnails from a directory."""
        self.clear()
        self._current_directory = directory

        if not directory.exists():
            return

        # Get image files
        image_files = []
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            image_files.extend(directory.glob(f"*{ext}"))

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

    def get_selected_path(self) -> Optional[Path]:
        """Get the currently selected image path."""
        return self._selected_path

    def select_path(self, path: Path):
        """Select a specific image by path."""
        self._on_thumbnail_clicked(path)

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
