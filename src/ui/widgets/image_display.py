"""Image display widget using Cairo rendering."""

from typing import Optional
from pathlib import Path
import io

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

from PIL import Image


class ImageDisplay(Gtk.DrawingArea):
    """Widget for displaying generated images using Cairo."""

    def __init__(self):
        super().__init__()
        self._pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self._pil_image: Optional[Image.Image] = None

        self.add_css_class("image-display")
        self.set_hexpand(True)
        self.set_vexpand(True)

        # Set draw function
        self.set_draw_func(self._on_draw)

    def _on_draw(self, area, cr, width, height):
        """Draw the image onto the widget."""
        # Draw background
        cr.set_source_rgb(0.1, 0.1, 0.1)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        if self._pixbuf is None:
            # Draw placeholder text
            cr.set_source_rgb(0.5, 0.5, 0.5)
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(16)

            text = "No image loaded"
            extents = cr.text_extents(text)
            x = (width - extents.width) / 2
            y = (height + extents.height) / 2

            cr.move_to(x, y)
            cr.show_text(text)
            return

        # Calculate scaling to fit while maintaining aspect ratio
        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        scale_x = width / img_width
        scale_y = height / img_height
        scale = min(scale_x, scale_y)

        # Calculate centered position
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        x = (width - scaled_width) / 2
        y = (height - scaled_height) / 2

        # Draw the image
        cr.save()
        cr.translate(x, y)
        cr.scale(scale, scale)

        Gdk.cairo_set_source_pixbuf(cr, self._pixbuf, 0, 0)
        cr.paint()

        cr.restore()

    def set_image(self, image: Image.Image):
        """Set the image to display from a PIL Image."""
        self._pil_image = image

        # Convert PIL Image to GdkPixbuf
        # Save to bytes buffer as PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Load into GdkPixbuf
        loader = GdkPixbuf.PixbufLoader.new_with_type("png")
        loader.write(buffer.read())
        loader.close()

        self._pixbuf = loader.get_pixbuf()

        # Trigger redraw
        self.queue_draw()

    def set_image_from_path(self, path: Path):
        """Set the image to display from a file path."""
        try:
            self._pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(path))
            self._pil_image = Image.open(path)
            self.queue_draw()
        except Exception as e:
            print(f"Error loading image from {path}: {e}")
            self.clear()

    def clear(self):
        """Clear the current image."""
        self._pixbuf = None
        self._pil_image = None
        self.queue_draw()

    def get_pil_image(self) -> Optional[Image.Image]:
        """Get the current image as PIL Image."""
        return self._pil_image

    def has_image(self) -> bool:
        """Check if an image is currently displayed."""
        return self._pixbuf is not None


class ImageDisplayFrame(Gtk.Frame):
    """Frame containing the image display with controls."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        """Build the widget UI."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(box)

        # Image display
        self._display = ImageDisplay()
        box.append(self._display)

    @property
    def display(self) -> ImageDisplay:
        """Get the image display widget."""
        return self._display

    def set_image(self, image: Image.Image):
        """Set the image to display."""
        self._display.set_image(image)

    def set_image_from_path(self, path: Path):
        """Set the image from a file path."""
        self._display.set_image_from_path(path)

    def clear(self):
        """Clear the image."""
        self._display.clear()

    def get_pil_image(self) -> Optional[Image.Image]:
        """Get the current PIL image."""
        return self._display.get_pil_image()

    def has_image(self) -> bool:
        """Check if an image is currently loaded."""
        return self._display.has_image()
