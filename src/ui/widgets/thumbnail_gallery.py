"""Thumbnail gallery widget for displaying generated images."""

from pathlib import Path
from typing import Optional, Callable, List
import io

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GdkPixbuf, GLib

from PIL import Image

from src.utils.constants import THUMBNAIL_SIZE, THUMBNAIL_COLUMNS


class ThumbnailItem(Gtk.Button):
    """Single thumbnail item in the gallery."""

    def __init__(
        self,
        path: Path,
        image: Optional[Image.Image] = None,
        on_click: Optional[Callable[[Path], None]] = None,
    ):
        super().__init__()
        self._path = path
        self._on_click = on_click

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
                    THUMBNAIL_SIZE,
                    THUMBNAIL_SIZE,
                    True,  # Preserve aspect ratio
                )
            except Exception:
                pixbuf = None

        if pixbuf:
            picture = Gtk.Picture.new_for_pixbuf(pixbuf)
            picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            picture.set_size_request(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            self.set_child(picture)
        else:
            # Placeholder
            label = Gtk.Label(label="?")
            label.set_size_request(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            self.set_child(label)

    def _pil_to_thumbnail(self, image: Image.Image) -> Optional[GdkPixbuf.Pixbuf]:
        """Convert PIL Image to thumbnail GdkPixbuf."""
        try:
            # Create thumbnail
            thumb = image.copy()
            thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))

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


class ThumbnailGallery(Gtk.Box):
    """Scrollable gallery of image thumbnails."""

    def __init__(
        self,
        on_image_selected: Optional[Callable[[Path], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._on_image_selected = on_image_selected
        self._thumbnails: List[ThumbnailItem] = []
        self._selected_path: Optional[Path] = None

        self.add_css_class("thumbnail-gallery")
        self._build_ui()

    def _build_ui(self):
        """Build the gallery UI."""
        # Header
        header = Gtk.Label(label="Generated Images")
        header.add_css_class("section-header")
        header.set_halign(Gtk.Align.START)
        self.append(header)

        # Scrolled window
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        self.append(scrolled)

        # Flow box for thumbnails
        self._flowbox = Gtk.FlowBox()
        self._flowbox.set_valign(Gtk.Align.START)
        self._flowbox.set_max_children_per_line(THUMBNAIL_COLUMNS)
        self._flowbox.set_min_children_per_line(1)
        self._flowbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self._flowbox.set_homogeneous(True)
        scrolled.set_child(self._flowbox)

    def add_image(self, path: Path, image: Optional[Image.Image] = None):
        """Add a new image to the gallery."""
        thumbnail = ThumbnailItem(
            path=path,
            image=image,
            on_click=self._on_thumbnail_clicked,
        )
        self._thumbnails.insert(0, thumbnail)

        # Add to flowbox at the beginning
        self._flowbox.prepend(thumbnail)

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
        self._selected_path = None

    def load_from_directory(self, directory: Path):
        """Load thumbnails from a directory."""
        self.clear()

        if not directory.exists():
            return

        # Get image files sorted by modification time (newest first)
        image_files = []
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            image_files.extend(directory.glob(f"*{ext}"))

        image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Add thumbnails
        for path in image_files:
            self.add_image(path)

    def get_selected_path(self) -> Optional[Path]:
        """Get the currently selected image path."""
        return self._selected_path

    def select_path(self, path: Path):
        """Select a specific image by path."""
        self._on_thumbnail_clicked(path)
