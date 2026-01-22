"""Image display widget using Cairo rendering with mask support."""

from typing import Optional, Callable
from pathlib import Path
from enum import Enum
import io

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

from PIL import Image, ImageDraw


class MaskTool(Enum):
    """Mask drawing tools."""
    NONE = "none"
    RECT = "rect"
    PAINT = "paint"


class ImageDisplay(Gtk.DrawingArea):
    """Widget for displaying generated images using Cairo with mask overlay support."""

    PAINT_RADIUS = 25  # Brush radius for paint tool
    MASK_COLOR = (144, 238, 144, 128)  # Light green with 50% transparency
    ZOOM_STEP = 1.2  # Zoom factor per scroll step
    MIN_ZOOM = 0.1  # Minimum zoom (10%)
    MAX_ZOOM = 20.0  # Maximum zoom (2000%)

    def __init__(self):
        super().__init__()
        self._pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self._pil_image: Optional[Image.Image] = None
        self._original_image: Optional[Image.Image] = None  # Store original for inpaint

        # Mask-related state
        self._mask_image: Optional[Image.Image] = None  # RGBA mask
        self._mask_pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self._inpaint_mode: bool = False
        self._current_tool: MaskTool = MaskTool.NONE

        # Drawing state
        self._is_drawing: bool = False
        self._draw_start_x: float = 0
        self._draw_start_y: float = 0
        self._current_x: float = 0
        self._current_y: float = 0

        # Transform cache (for coordinate conversion)
        self._img_x: float = 0
        self._img_y: float = 0
        self._scale: float = 1.0

        # Zoom and pan state
        self._zoom_level: float = 1.0  # 1.0 = fit to window
        self._pan_x: float = 0  # Pan offset in image pixels
        self._pan_y: float = 0
        self._is_panning: bool = False
        self._pan_start_x: float = 0
        self._pan_start_y: float = 0

        self.add_css_class("image-display")
        self.set_hexpand(True)
        self.set_vexpand(True)

        # Make focusable for keyboard events
        self.set_focusable(True)

        # Set draw function
        self.set_draw_func(self._on_draw)

        # Set up event controllers
        self._setup_event_controllers()

    def _setup_event_controllers(self):
        """Set up mouse event controllers."""
        # Click/drag controller
        click_controller = Gtk.GestureClick()
        click_controller.connect("pressed", self._on_mouse_pressed)
        click_controller.connect("released", self._on_mouse_released)
        self.add_controller(click_controller)

        # Middle-click for panning
        pan_click_controller = Gtk.GestureClick()
        pan_click_controller.set_button(2)  # Middle mouse button
        pan_click_controller.connect("pressed", self._on_pan_pressed)
        pan_click_controller.connect("released", self._on_pan_released)
        self.add_controller(pan_click_controller)

        # Drag controller for continuous drawing
        drag_controller = Gtk.GestureDrag()
        drag_controller.connect("drag-begin", self._on_drag_begin)
        drag_controller.connect("drag-update", self._on_drag_update)
        drag_controller.connect("drag-end", self._on_drag_end)
        self.add_controller(drag_controller)

        # Motion controller for cursor updates
        motion_controller = Gtk.EventControllerMotion()
        motion_controller.connect("motion", self._on_motion)
        motion_controller.connect("enter", self._on_enter)
        motion_controller.connect("leave", self._on_leave)
        self.add_controller(motion_controller)

        # Scroll controller for zoom
        scroll_controller = Gtk.EventControllerScroll()
        scroll_controller.set_flags(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll_controller.connect("scroll", self._on_scroll)
        self.add_controller(scroll_controller)

        # Key controller for keyboard zoom
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _widget_to_image_coords(self, widget_x: float, widget_y: float) -> tuple:
        """Convert widget coordinates to image coordinates."""
        if self._pixbuf is None or self._scale == 0:
            return (-1, -1)

        # Reverse the transform from _on_draw (accounts for zoom and pan)
        img_x = (widget_x - self._img_x) / self._scale + self._pan_x
        img_y = (widget_y - self._img_y) / self._scale + self._pan_y

        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        # Clamp to image bounds
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))

        return (int(img_x), int(img_y))

    def _on_motion(self, controller, x, y):
        """Handle mouse motion for cursor updates and panning."""
        # Handle panning with middle mouse button
        if self._is_panning and self._pixbuf is not None:
            # Calculate pan delta in image coordinates
            dx = x - self._pan_start_x
            dy = y - self._pan_start_y

            # Update pan (convert screen delta to image delta)
            self._pan_x -= dx / self._scale
            self._pan_y -= dy / self._scale

            # Clamp pan
            self._clamp_pan()

            # Update start position for next motion
            self._pan_start_x = x
            self._pan_start_y = y

            self.queue_draw()

        self._current_x = x
        self._current_y = y

        # Update cursor based on tool
        if self._is_panning:
            self.set_cursor(Gdk.Cursor.new_from_name("grabbing", None))
        elif self._inpaint_mode and self._current_tool != MaskTool.NONE:
            self.set_cursor(Gdk.Cursor.new_from_name("crosshair", None))
        else:
            self.set_cursor(None)

        # Redraw for live preview of paint cursor
        if self._inpaint_mode and self._current_tool == MaskTool.PAINT:
            self.queue_draw()

    def _on_enter(self, controller, x, y):
        """Handle mouse enter."""
        # Grab focus for keyboard events
        self.grab_focus()
        self._on_motion(controller, x, y)

    def _on_leave(self, controller):
        """Handle mouse leave."""
        self.set_cursor(None)
        self.queue_draw()

    def _on_scroll(self, controller, dx, dy):
        """Handle scroll for zoom."""
        if self._pixbuf is None:
            return False

        # Get cursor position for zoom center
        cursor_x = self._current_x
        cursor_y = self._current_y

        # Calculate zoom factor
        if dy < 0:
            # Scroll up = zoom in
            factor = self.ZOOM_STEP
        else:
            # Scroll down = zoom out
            factor = 1.0 / self.ZOOM_STEP

        self._zoom_at_point(cursor_x, cursor_y, factor)
        return True

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle keyboard shortcuts for zoom."""
        if self._pixbuf is None:
            return False

        # Check for Ctrl modifier
        ctrl_pressed = state & Gdk.ModifierType.CONTROL_MASK

        if ctrl_pressed:
            # Ctrl+Plus or Ctrl+Equal for zoom in
            if keyval in (Gdk.KEY_plus, Gdk.KEY_equal, Gdk.KEY_KP_Add):
                self._zoom_at_point(
                    self.get_width() / 2,
                    self.get_height() / 2,
                    self.ZOOM_STEP
                )
                return True
            # Ctrl+Minus for zoom out
            elif keyval in (Gdk.KEY_minus, Gdk.KEY_KP_Subtract):
                self._zoom_at_point(
                    self.get_width() / 2,
                    self.get_height() / 2,
                    1.0 / self.ZOOM_STEP
                )
                return True
            # Ctrl+0 for reset zoom
            elif keyval in (Gdk.KEY_0, Gdk.KEY_KP_0):
                self.reset_zoom()
                return True

        # Also allow 'r' or 'R' to reset zoom (no modifier needed)
        if keyval in (Gdk.KEY_r, Gdk.KEY_R):
            self.reset_zoom()
            return True

        # Also allow '0' without Ctrl to reset zoom
        if keyval in (Gdk.KEY_0, Gdk.KEY_KP_0):
            self.reset_zoom()
            return True

        return False

    def _on_pan_pressed(self, gesture, n_press, x, y):
        """Handle middle mouse button press for panning."""
        if self._pixbuf is None:
            return

        self._is_panning = True
        self._pan_start_x = x
        self._pan_start_y = y
        self.set_cursor(Gdk.Cursor.new_from_name("grabbing", None))

    def _on_pan_released(self, gesture, n_press, x, y):
        """Handle middle mouse button release."""
        self._is_panning = False
        self.set_cursor(None)

    def _zoom_at_point(self, widget_x: float, widget_y: float, factor: float):
        """Zoom centered on a specific point in widget coordinates."""
        if self._pixbuf is None:
            return

        # Calculate new zoom level
        old_zoom = self._zoom_level
        new_zoom = old_zoom * factor

        # Clamp zoom level
        new_zoom = max(self.MIN_ZOOM, min(new_zoom, self.MAX_ZOOM))

        if new_zoom == old_zoom:
            return

        # Get the point in image coordinates before zoom
        # First, get the base scale (fit to window)
        widget_width = self.get_width()
        widget_height = self.get_height()
        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        base_scale_x = widget_width / img_width
        base_scale_y = widget_height / img_height
        base_scale = min(base_scale_x, base_scale_y)

        # Current effective scale
        old_effective_scale = base_scale * old_zoom
        new_effective_scale = base_scale * new_zoom

        # Point in image coordinates (relative to pan offset)
        # img_point = (widget_point - img_offset) / scale + pan_offset
        old_img_x = (widget_x - self._img_x) / old_effective_scale + self._pan_x
        old_img_y = (widget_y - self._img_y) / old_effective_scale + self._pan_y

        # Update zoom
        self._zoom_level = new_zoom

        # Calculate new scaled dimensions
        new_scaled_width = img_width * new_effective_scale
        new_scaled_height = img_height * new_effective_scale

        # Calculate new image offset (centered)
        new_img_x = (widget_width - new_scaled_width) / 2
        new_img_y = (widget_height - new_scaled_height) / 2

        # Calculate new pan to keep the same image point under cursor
        # widget_point = (img_point - pan_offset) * scale + img_offset
        # => pan_offset = img_point - (widget_point - img_offset) / scale
        self._pan_x = old_img_x - (widget_x - new_img_x) / new_effective_scale
        self._pan_y = old_img_y - (widget_y - new_img_y) / new_effective_scale

        # Clamp pan to reasonable bounds
        self._clamp_pan()

        self.queue_draw()

    def _clamp_pan(self):
        """Clamp pan values to prevent excessive panning."""
        if self._pixbuf is None:
            return

        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        # Allow panning but keep at least some of the image visible
        max_pan_x = img_width * 0.9
        max_pan_y = img_height * 0.9

        self._pan_x = max(-max_pan_x, min(self._pan_x, max_pan_x))
        self._pan_y = max(-max_pan_y, min(self._pan_y, max_pan_y))

    def reset_zoom(self):
        """Reset zoom to fit window and center image."""
        self._zoom_level = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self.queue_draw()

    def _on_mouse_pressed(self, gesture, n_press, x, y):
        """Handle mouse button press."""
        # Grab focus for keyboard events
        self.grab_focus()

        if not self._inpaint_mode or self._current_tool == MaskTool.NONE:
            return

        if self._pixbuf is None:
            return

        self._is_drawing = True
        self._draw_start_x = x
        self._draw_start_y = y

        # Ensure mask exists
        self._ensure_mask()

        # For paint tool, start painting immediately
        if self._current_tool == MaskTool.PAINT:
            img_x, img_y = self._widget_to_image_coords(x, y)
            self._paint_at(img_x, img_y)

    def _on_mouse_released(self, gesture, n_press, x, y):
        """Handle mouse button release."""
        if not self._is_drawing:
            return

        self._is_drawing = False

        # For rect tool, finalize the rectangle
        if self._current_tool == MaskTool.RECT:
            self._finalize_rect(x, y)

        self.queue_draw()

    def _on_drag_begin(self, gesture, start_x, start_y):
        """Handle drag start."""
        if not self._inpaint_mode or self._current_tool == MaskTool.NONE:
            return

        self._is_drawing = True
        self._draw_start_x = start_x
        self._draw_start_y = start_y

        self._ensure_mask()

    def _on_drag_update(self, gesture, offset_x, offset_y):
        """Handle drag update."""
        if not self._is_drawing:
            return

        current_x = self._draw_start_x + offset_x
        current_y = self._draw_start_y + offset_y

        if self._current_tool == MaskTool.PAINT:
            img_x, img_y = self._widget_to_image_coords(current_x, current_y)
            self._paint_at(img_x, img_y)
        elif self._current_tool == MaskTool.RECT:
            # Just update for preview
            self._current_x = current_x
            self._current_y = current_y
            self.queue_draw()

    def _on_drag_end(self, gesture, offset_x, offset_y):
        """Handle drag end."""
        if not self._is_drawing:
            return

        self._is_drawing = False

        if self._current_tool == MaskTool.RECT:
            end_x = self._draw_start_x + offset_x
            end_y = self._draw_start_y + offset_y
            self._finalize_rect(end_x, end_y)

    def _ensure_mask(self):
        """Ensure mask image exists and is correct size."""
        if self._pixbuf is None:
            return

        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        if self._mask_image is None or \
           self._mask_image.width != img_width or \
           self._mask_image.height != img_height:
            # Create new RGBA mask (transparent)
            self._mask_image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
            self._update_mask_pixbuf()

    def _paint_at(self, img_x: int, img_y: int):
        """Paint a circle at the given image coordinates."""
        if self._mask_image is None:
            return

        draw = ImageDraw.Draw(self._mask_image)
        r = self.PAINT_RADIUS
        draw.ellipse(
            [img_x - r, img_y - r, img_x + r, img_y + r],
            fill=self.MASK_COLOR
        )
        self._update_mask_pixbuf()
        self.queue_draw()

    def _finalize_rect(self, end_x: float, end_y: float):
        """Finalize drawing a rectangle mask."""
        if self._mask_image is None:
            return

        # Convert to image coordinates
        start_img_x, start_img_y = self._widget_to_image_coords(self._draw_start_x, self._draw_start_y)
        end_img_x, end_img_y = self._widget_to_image_coords(end_x, end_y)

        # Ensure proper ordering
        x1, x2 = min(start_img_x, end_img_x), max(start_img_x, end_img_x)
        y1, y2 = min(start_img_y, end_img_y), max(start_img_y, end_img_y)

        # Draw filled rectangle
        draw = ImageDraw.Draw(self._mask_image)
        draw.rectangle([x1, y1, x2, y2], fill=self.MASK_COLOR)
        self._update_mask_pixbuf()
        self.queue_draw()

    def _update_mask_pixbuf(self):
        """Update the mask pixbuf from the mask image."""
        if self._mask_image is None:
            self._mask_pixbuf = None
            return

        # Convert PIL Image to GdkPixbuf
        buffer = io.BytesIO()
        self._mask_image.save(buffer, format="PNG")
        buffer.seek(0)

        loader = GdkPixbuf.PixbufLoader.new_with_type("png")
        loader.write(buffer.read())
        loader.close()

        self._mask_pixbuf = loader.get_pixbuf()

    def _on_draw(self, area, cr, width, height):
        """Draw the image and mask onto the widget."""
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

        # Calculate base scaling to fit while maintaining aspect ratio
        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        base_scale_x = width / img_width
        base_scale_y = height / img_height
        base_scale = min(base_scale_x, base_scale_y)

        # Apply zoom level
        self._scale = base_scale * self._zoom_level

        # Calculate centered position
        scaled_width = img_width * self._scale
        scaled_height = img_height * self._scale
        self._img_x = (width - scaled_width) / 2
        self._img_y = (height - scaled_height) / 2

        # Draw the image with pan offset
        cr.save()
        cr.translate(self._img_x, self._img_y)
        cr.scale(self._scale, self._scale)
        cr.translate(-self._pan_x, -self._pan_y)

        Gdk.cairo_set_source_pixbuf(cr, self._pixbuf, 0, 0)
        cr.paint()

        # Draw the mask overlay
        if self._mask_pixbuf is not None and self._inpaint_mode:
            Gdk.cairo_set_source_pixbuf(cr, self._mask_pixbuf, 0, 0)
            cr.paint()

        cr.restore()

        # Draw zoom indicator
        if self._zoom_level != 1.0:
            self._draw_zoom_indicator(cr, width, height)

        # Draw rect preview while dragging
        if self._is_drawing and self._current_tool == MaskTool.RECT:
            self._draw_rect_preview(cr)

        # Draw paint cursor preview
        if self._inpaint_mode and self._current_tool == MaskTool.PAINT and not self._is_drawing:
            self._draw_paint_cursor(cr)

    def _draw_zoom_indicator(self, cr, width, height):
        """Draw zoom level indicator."""
        zoom_percent = int(self._zoom_level * 100)
        text = f"{zoom_percent}%"

        cr.set_source_rgba(0.0, 0.0, 0.0, 0.6)
        cr.rectangle(width - 70, 10, 60, 25)
        cr.fill()

        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(12)
        extents = cr.text_extents(text)
        cr.move_to(width - 40 - extents.width / 2, 27)
        cr.show_text(text)

    def _draw_rect_preview(self, cr):
        """Draw rectangle preview while dragging."""
        x1, y1 = self._draw_start_x, self._draw_start_y
        x2, y2 = self._current_x, self._current_y

        # Ensure proper ordering
        left = min(x1, x2)
        top = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        # Draw dashed outline
        cr.set_source_rgba(0.5, 0.9, 0.5, 0.8)
        cr.set_line_width(2)
        cr.set_dash([5, 5])
        cr.rectangle(left, top, w, h)
        cr.stroke()

        # Draw semi-transparent fill
        cr.set_source_rgba(0.5, 0.9, 0.5, 0.3)
        cr.rectangle(left, top, w, h)
        cr.fill()

    def _draw_paint_cursor(self, cr):
        """Draw paint brush cursor preview."""
        # Convert cursor position to screen radius
        scaled_radius = self.PAINT_RADIUS * self._scale

        cr.set_source_rgba(0.5, 0.9, 0.5, 0.6)
        cr.set_line_width(2)
        cr.arc(self._current_x, self._current_y, scaled_radius, 0, 2 * 3.14159)
        cr.stroke()

    def set_image(self, image: Image.Image):
        """Set the image to display from a PIL Image."""
        self._pil_image = image

        # Convert PIL Image to GdkPixbuf
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        loader = GdkPixbuf.PixbufLoader.new_with_type("png")
        loader.write(buffer.read())
        loader.close()

        self._pixbuf = loader.get_pixbuf()

        # Clear mask when setting new image (unless in inpaint mode with original)
        if not self._inpaint_mode:
            self._mask_image = None
            self._mask_pixbuf = None
            # Reset zoom when loading new image (not in inpaint mode)
            self.reset_zoom()
        else:
            # Trigger redraw
            self.queue_draw()

    def set_image_from_path(self, path: Path):
        """Set the image to display from a file path."""
        try:
            self._pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(path))
            self._pil_image = Image.open(path).convert("RGB")
            # Clear mask
            self._mask_image = None
            self._mask_pixbuf = None
            # Reset zoom when loading new image
            self.reset_zoom()
        except Exception as e:
            print(f"Error loading image from {path}: {e}")
            self.clear()

    def clear(self):
        """Clear the current image."""
        self._pixbuf = None
        self._pil_image = None
        self._original_image = None
        self._mask_image = None
        self._mask_pixbuf = None
        # Reset zoom
        self._zoom_level = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self.queue_draw()

    def get_pil_image(self) -> Optional[Image.Image]:
        """Get the current image as PIL Image."""
        return self._pil_image

    def has_image(self) -> bool:
        """Check if an image is currently displayed."""
        return self._pixbuf is not None

    # Inpaint mode methods
    def set_inpaint_mode(self, enabled: bool):
        """Enable or disable inpaint mode."""
        self._inpaint_mode = enabled
        if enabled:
            # Store original image when entering inpaint mode
            if self._pil_image is not None:
                self._original_image = self._pil_image.copy()
        else:
            # Clear mask and restore original when exiting
            self._mask_image = None
            self._mask_pixbuf = None
            self._original_image = None
            self._current_tool = MaskTool.NONE
        self.queue_draw()

    def set_mask_tool(self, tool: MaskTool):
        """Set the current mask drawing tool."""
        self._current_tool = tool
        self.queue_draw()

    def clear_masks(self):
        """Clear all drawn masks."""
        if self._mask_image is not None:
            self._mask_image = Image.new(
                "RGBA",
                (self._mask_image.width, self._mask_image.height),
                (0, 0, 0, 0)
            )
            self._update_mask_pixbuf()
            self.queue_draw()

    def get_mask_image(self) -> Optional[Image.Image]:
        """Get the current mask as a PIL Image (white = mask, black = no mask)."""
        if self._mask_image is None:
            return None

        # Convert RGBA mask to binary mask (white where alpha > 0)
        mask = Image.new("L", self._mask_image.size, 0)
        for x in range(self._mask_image.width):
            for y in range(self._mask_image.height):
                r, g, b, a = self._mask_image.getpixel((x, y))
                if a > 0:
                    mask.putpixel((x, y), 255)
        return mask

    def get_mask_image_fast(self) -> Optional[Image.Image]:
        """Get the current mask as a PIL Image (faster method using numpy if available)."""
        if self._mask_image is None:
            return None

        try:
            import numpy as np
            # Convert to numpy for fast processing
            arr = np.array(self._mask_image)
            # Alpha channel > 0 means masked
            mask_arr = (arr[:, :, 3] > 0).astype(np.uint8) * 255
            return Image.fromarray(mask_arr, mode="L")
        except ImportError:
            # Fall back to slow method
            return self.get_mask_image()

    def get_original_image(self) -> Optional[Image.Image]:
        """Get the original image (before inpaint generation)."""
        return self._original_image

    def has_mask(self) -> bool:
        """Check if any mask has been drawn."""
        if self._mask_image is None:
            return False
        # Check if any pixel has alpha > 0
        try:
            import numpy as np
            arr = np.array(self._mask_image)
            return np.any(arr[:, :, 3] > 0)
        except ImportError:
            # Slow method
            for x in range(self._mask_image.width):
                for y in range(self._mask_image.height):
                    if self._mask_image.getpixel((x, y))[3] > 0:
                        return True
            return False

    @property
    def inpaint_mode(self) -> bool:
        """Check if inpaint mode is active."""
        return self._inpaint_mode


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

    # Inpaint mode methods (delegate to display)
    def set_inpaint_mode(self, enabled: bool):
        """Enable or disable inpaint mode."""
        self._display.set_inpaint_mode(enabled)

    def set_mask_tool(self, tool: MaskTool):
        """Set the current mask drawing tool."""
        self._display.set_mask_tool(tool)

    def clear_masks(self):
        """Clear all drawn masks."""
        self._display.clear_masks()

    def get_mask_image(self) -> Optional[Image.Image]:
        """Get the current mask as a PIL Image."""
        return self._display.get_mask_image_fast()

    def get_original_image(self) -> Optional[Image.Image]:
        """Get the original image (before inpaint generation)."""
        return self._display.get_original_image()

    def has_mask(self) -> bool:
        """Check if any mask has been drawn."""
        return self._display.has_mask()

    @property
    def inpaint_mode(self) -> bool:
        """Check if inpaint mode is active."""
        return self._display.inpaint_mode

    def reset_zoom(self):
        """Reset zoom to fit window."""
        self._display.reset_zoom()
