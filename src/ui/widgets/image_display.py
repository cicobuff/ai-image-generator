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


class OutpaintTool(Enum):
    """Outpaint mask drawing tools."""
    NONE = "none"
    EDGE = "edge"


class OutpaintDirection(Enum):
    """Direction for outpaint extension."""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class ImageDisplay(Gtk.DrawingArea):
    """Widget for displaying generated images using Cairo with mask overlay support."""

    PAINT_RADIUS = 25  # Brush radius for paint tool
    MASK_COLOR = (144, 238, 144, 128)  # Light green with 50% transparency
    OUTPAINT_MASK_COLOR = (236, 64, 122, 128)  # Pink with 50% transparency
    ZOOM_STEP = 1.2  # Zoom factor per scroll step
    MIN_ZOOM = 0.1  # Minimum zoom (10%)
    MAX_ZOOM = 20.0  # Maximum zoom (2000%)
    DEFAULT_EDGE_ZONE = 200  # Default edge zone size in pixels

    def __init__(self, edge_zone_size: int = DEFAULT_EDGE_ZONE):
        super().__init__()
        self._pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self._pil_image: Optional[Image.Image] = None
        self._original_image: Optional[Image.Image] = None  # Store original for inpaint

        # Mask-related state (inpaint)
        self._mask_image: Optional[Image.Image] = None  # RGBA mask
        self._mask_pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self._inpaint_mode: bool = False
        self._current_tool: MaskTool = MaskTool.NONE

        # Outpaint-related state
        self._outpaint_mode: bool = False
        self._outpaint_tool: OutpaintTool = OutpaintTool.NONE
        self._edge_zone_size: int = edge_zone_size
        # Store extension amounts for each edge (in pixels)
        self._outpaint_extensions: dict = {
            OutpaintDirection.LEFT: 0,
            OutpaintDirection.RIGHT: 0,
            OutpaintDirection.TOP: 0,
            OutpaintDirection.BOTTOM: 0,
        }
        self._current_outpaint_direction: OutpaintDirection = OutpaintDirection.NONE
        self._outpaint_draw_start: float = 0  # Starting position for outpaint drag

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

    def _widget_to_image_coords_unclamped(self, widget_x: float, widget_y: float) -> tuple:
        """Convert widget coordinates to image coordinates without clamping."""
        if self._pixbuf is None or self._scale == 0:
            return (-1, -1)

        img_x = (widget_x - self._img_x) / self._scale + self._pan_x
        img_y = (widget_y - self._img_y) / self._scale + self._pan_y

        return (img_x, img_y)

    def _get_outpaint_direction(self, widget_x: float, widget_y: float) -> OutpaintDirection:
        """Determine which edge zone the cursor is in for outpainting.

        Returns OutpaintDirection.NONE if not in a valid edge zone.
        Corners are ignored (return NONE).
        """
        if self._pixbuf is None:
            return OutpaintDirection.NONE

        img_x, img_y = self._widget_to_image_coords_unclamped(widget_x, widget_y)
        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        # Check if within image bounds (for edge detection)
        in_image_x = 0 <= img_x < img_width
        in_image_y = 0 <= img_y < img_height

        if not in_image_x or not in_image_y:
            return OutpaintDirection.NONE

        edge_zone = self._edge_zone_size

        # Check corners first - they are ignored
        in_left_zone = img_x < edge_zone
        in_right_zone = img_x >= img_width - edge_zone
        in_top_zone = img_y < edge_zone
        in_bottom_zone = img_y >= img_height - edge_zone

        # Corner check: if in two edge zones at once, ignore
        edge_count = sum([in_left_zone, in_right_zone, in_top_zone, in_bottom_zone])
        if edge_count >= 2:
            return OutpaintDirection.NONE

        # Determine direction
        if in_left_zone:
            return OutpaintDirection.LEFT
        elif in_right_zone:
            return OutpaintDirection.RIGHT
        elif in_top_zone:
            return OutpaintDirection.TOP
        elif in_bottom_zone:
            return OutpaintDirection.BOTTOM

        return OutpaintDirection.NONE

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

        # Update cursor based on mode and tool
        if self._is_panning:
            self.set_cursor(Gdk.Cursor.new_from_name("grabbing", None))
        elif self._outpaint_mode and self._outpaint_tool == OutpaintTool.EDGE:
            # Determine which edge zone we're in and set appropriate cursor
            direction = self._get_outpaint_direction(x, y)
            if direction == OutpaintDirection.LEFT:
                self.set_cursor(Gdk.Cursor.new_from_name("w-resize", None))
            elif direction == OutpaintDirection.RIGHT:
                self.set_cursor(Gdk.Cursor.new_from_name("e-resize", None))
            elif direction == OutpaintDirection.TOP:
                self.set_cursor(Gdk.Cursor.new_from_name("n-resize", None))
            elif direction == OutpaintDirection.BOTTOM:
                self.set_cursor(Gdk.Cursor.new_from_name("s-resize", None))
            else:
                self.set_cursor(Gdk.Cursor.new_from_name("not-allowed", None))
        elif self._inpaint_mode and self._current_tool != MaskTool.NONE:
            self.set_cursor(Gdk.Cursor.new_from_name("crosshair", None))
        else:
            self.set_cursor(None)

        # Redraw for live preview of paint cursor or outpaint zones
        if self._inpaint_mode and self._current_tool == MaskTool.PAINT:
            self.queue_draw()
        elif self._outpaint_mode and self._outpaint_tool == OutpaintTool.EDGE:
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

        # Handle outpaint mode
        if self._outpaint_mode and self._outpaint_tool == OutpaintTool.EDGE:
            if self._pixbuf is None:
                return

            direction = self._get_outpaint_direction(x, y)
            if direction == OutpaintDirection.NONE:
                return  # Not in a valid edge zone

            self._is_drawing = True
            self._current_outpaint_direction = direction
            self._draw_start_x = x
            self._draw_start_y = y

            # Store the starting image coordinate based on direction
            img_x, img_y = self._widget_to_image_coords(x, y)
            if direction in (OutpaintDirection.LEFT, OutpaintDirection.RIGHT):
                self._outpaint_draw_start = img_x
            else:
                self._outpaint_draw_start = img_y
            return

        # Handle inpaint mode
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
        # Handle outpaint mode
        if self._outpaint_mode and self._outpaint_tool == OutpaintTool.EDGE:
            if self._pixbuf is None:
                return

            direction = self._get_outpaint_direction(start_x, start_y)
            if direction == OutpaintDirection.NONE:
                return

            self._is_drawing = True
            self._current_outpaint_direction = direction
            self._draw_start_x = start_x
            self._draw_start_y = start_y

            img_x, img_y = self._widget_to_image_coords(start_x, start_y)
            if direction in (OutpaintDirection.LEFT, OutpaintDirection.RIGHT):
                self._outpaint_draw_start = img_x
            else:
                self._outpaint_draw_start = img_y
            return

        # Handle inpaint mode
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

        # Handle outpaint mode
        if self._outpaint_mode and self._current_outpaint_direction != OutpaintDirection.NONE:
            img_x, img_y = self._widget_to_image_coords_unclamped(current_x, current_y)
            img_width = self._pixbuf.get_width()
            img_height = self._pixbuf.get_height()

            # Calculate extension based on drag direction
            direction = self._current_outpaint_direction
            if direction == OutpaintDirection.LEFT:
                # Drag left (negative x direction) creates extension
                extension = max(0, int(self._outpaint_draw_start - img_x))
            elif direction == OutpaintDirection.RIGHT:
                # Drag right (positive x direction) creates extension
                extension = max(0, int(img_x - self._outpaint_draw_start))
            elif direction == OutpaintDirection.TOP:
                # Drag up (negative y direction) creates extension
                extension = max(0, int(self._outpaint_draw_start - img_y))
            elif direction == OutpaintDirection.BOTTOM:
                # Drag down (positive y direction) creates extension
                extension = max(0, int(img_y - self._outpaint_draw_start))
            else:
                extension = 0

            # Update extension (live preview)
            self._outpaint_extensions[direction] = extension
            self._current_x = current_x
            self._current_y = current_y
            self.queue_draw()
            return

        # Handle inpaint mode
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

        # Handle outpaint mode - extension is already set during drag
        if self._outpaint_mode and self._current_outpaint_direction != OutpaintDirection.NONE:
            # Extension is finalized, reset drawing state
            self._current_outpaint_direction = OutpaintDirection.NONE
            self.queue_draw()
            return

        # Handle inpaint mode
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

        # Draw the inpaint mask overlay
        if self._mask_pixbuf is not None and self._inpaint_mode:
            Gdk.cairo_set_source_pixbuf(cr, self._mask_pixbuf, 0, 0)
            cr.paint()

        cr.restore()

        # Draw outpaint extension overlays
        if self._outpaint_mode:
            self._draw_outpaint_overlays(cr, width, height)

        # Draw outpaint edge zone indicators when in outpaint mode
        if self._outpaint_mode and self._outpaint_tool == OutpaintTool.EDGE and not self._is_drawing:
            self._draw_outpaint_edge_zones(cr, width, height)

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

    def _draw_outpaint_overlays(self, cr, width, height):
        """Draw outpaint extension overlays showing where the image will be extended."""
        if self._pixbuf is None:
            return

        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()

        # Pink color for outpaint (r, g, b, a normalized)
        pink = (236/255, 64/255, 122/255)

        for direction, extension in self._outpaint_extensions.items():
            if extension <= 0:
                continue

            # Calculate screen coordinates for the extension area
            if direction == OutpaintDirection.LEFT:
                # Extension to the left of the image
                screen_x = self._img_x - extension * self._scale - self._pan_x * self._scale
                screen_y = self._img_y - self._pan_y * self._scale
                screen_w = extension * self._scale
                screen_h = img_height * self._scale
            elif direction == OutpaintDirection.RIGHT:
                # Extension to the right of the image
                screen_x = self._img_x + (img_width - self._pan_x) * self._scale
                screen_y = self._img_y - self._pan_y * self._scale
                screen_w = extension * self._scale
                screen_h = img_height * self._scale
            elif direction == OutpaintDirection.TOP:
                # Extension above the image
                screen_x = self._img_x - self._pan_x * self._scale
                screen_y = self._img_y - extension * self._scale - self._pan_y * self._scale
                screen_w = img_width * self._scale
                screen_h = extension * self._scale
            elif direction == OutpaintDirection.BOTTOM:
                # Extension below the image
                screen_x = self._img_x - self._pan_x * self._scale
                screen_y = self._img_y + (img_height - self._pan_y) * self._scale
                screen_w = img_width * self._scale
                screen_h = extension * self._scale
            else:
                continue

            # Draw semi-transparent pink fill
            cr.set_source_rgba(pink[0], pink[1], pink[2], 0.4)
            cr.rectangle(screen_x, screen_y, screen_w, screen_h)
            cr.fill()

            # Draw dashed outline
            cr.set_source_rgba(pink[0], pink[1], pink[2], 0.9)
            cr.set_line_width(2)
            cr.set_dash([5, 5])
            cr.rectangle(screen_x, screen_y, screen_w, screen_h)
            cr.stroke()
            cr.set_dash([])  # Reset dash

            # Draw extension amount label
            label = f"{extension}px"
            cr.set_source_rgba(1, 1, 1, 0.9)
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(12)
            extents = cr.text_extents(label)

            label_x = screen_x + screen_w / 2 - extents.width / 2
            label_y = screen_y + screen_h / 2 + extents.height / 2

            # Draw text background
            cr.set_source_rgba(0, 0, 0, 0.6)
            cr.rectangle(label_x - 4, label_y - extents.height - 2, extents.width + 8, extents.height + 4)
            cr.fill()

            # Draw text
            cr.set_source_rgba(1, 1, 1, 0.9)
            cr.move_to(label_x, label_y)
            cr.show_text(label)

    def _draw_outpaint_edge_zones(self, cr, width, height):
        """Draw indicators for the active edge zones where outpaint can be started."""
        if self._pixbuf is None:
            return

        img_width = self._pixbuf.get_width()
        img_height = self._pixbuf.get_height()
        edge_zone = self._edge_zone_size

        # Get current cursor position to highlight the active zone
        current_direction = self._get_outpaint_direction(self._current_x, self._current_y)

        # Pink color
        pink = (236/255, 64/255, 122/255)

        # Draw subtle edge zone indicators
        zones = [
            (OutpaintDirection.LEFT, 0, 0, edge_zone, img_height),
            (OutpaintDirection.RIGHT, img_width - edge_zone, 0, edge_zone, img_height),
            (OutpaintDirection.TOP, 0, 0, img_width, edge_zone),
            (OutpaintDirection.BOTTOM, 0, img_height - edge_zone, img_width, edge_zone),
        ]

        for direction, x, y, w, h in zones:
            # Convert to screen coordinates
            screen_x = self._img_x + (x - self._pan_x) * self._scale
            screen_y = self._img_y + (y - self._pan_y) * self._scale
            screen_w = w * self._scale
            screen_h = h * self._scale

            # Highlight the zone under the cursor more prominently
            if direction == current_direction:
                cr.set_source_rgba(pink[0], pink[1], pink[2], 0.25)
            else:
                cr.set_source_rgba(pink[0], pink[1], pink[2], 0.08)

            cr.rectangle(screen_x, screen_y, screen_w, screen_h)
            cr.fill()

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

    # Outpaint mode methods
    def set_outpaint_mode(self, enabled: bool):
        """Enable or disable outpaint mode."""
        self._outpaint_mode = enabled
        if enabled:
            # Store original image when entering outpaint mode
            if self._pil_image is not None:
                self._original_image = self._pil_image.copy()
        else:
            # Clear outpaint extensions when exiting
            self._clear_outpaint_extensions()
            self._original_image = None
            self._outpaint_tool = OutpaintTool.NONE
        self.queue_draw()

    def set_outpaint_tool(self, tool: OutpaintTool):
        """Set the current outpaint drawing tool."""
        self._outpaint_tool = tool
        self.queue_draw()

    def _clear_outpaint_extensions(self):
        """Clear all outpaint extensions."""
        self._outpaint_extensions = {
            OutpaintDirection.LEFT: 0,
            OutpaintDirection.RIGHT: 0,
            OutpaintDirection.TOP: 0,
            OutpaintDirection.BOTTOM: 0,
        }

    def clear_outpaint_masks(self):
        """Clear all outpaint extension masks."""
        self._clear_outpaint_extensions()
        self.queue_draw()

    def get_outpaint_extensions(self) -> dict:
        """Get the current outpaint extension amounts.

        Returns:
            Dict with keys 'left', 'right', 'top', 'bottom' containing extension pixels.
        """
        return {
            'left': self._outpaint_extensions[OutpaintDirection.LEFT],
            'right': self._outpaint_extensions[OutpaintDirection.RIGHT],
            'top': self._outpaint_extensions[OutpaintDirection.TOP],
            'bottom': self._outpaint_extensions[OutpaintDirection.BOTTOM],
        }

    def has_outpaint_extensions(self) -> bool:
        """Check if any outpaint extensions have been defined."""
        return any(ext > 0 for ext in self._outpaint_extensions.values())

    @property
    def outpaint_mode(self) -> bool:
        """Check if outpaint mode is active."""
        return self._outpaint_mode

    def set_edge_zone_size(self, size: int):
        """Set the edge zone size for outpaint detection."""
        self._edge_zone_size = max(50, min(size, 500))  # Clamp between 50-500


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

    # Outpaint mode methods (delegate to display)
    def set_outpaint_mode(self, enabled: bool):
        """Enable or disable outpaint mode."""
        self._display.set_outpaint_mode(enabled)

    def set_outpaint_tool(self, tool: OutpaintTool):
        """Set the current outpaint drawing tool."""
        self._display.set_outpaint_tool(tool)

    def clear_outpaint_masks(self):
        """Clear all outpaint extension masks."""
        self._display.clear_outpaint_masks()

    def get_outpaint_extensions(self) -> dict:
        """Get the current outpaint extension amounts."""
        return self._display.get_outpaint_extensions()

    def has_outpaint_extensions(self) -> bool:
        """Check if any outpaint extensions have been defined."""
        return self._display.has_outpaint_extensions()

    @property
    def outpaint_mode(self) -> bool:
        """Check if outpaint mode is active."""
        return self._display.outpaint_mode

    def set_edge_zone_size(self, size: int):
        """Set the edge zone size for outpaint detection."""
        self._display.set_edge_zone_size(size)

    def reset_zoom(self):
        """Reset zoom to fit window."""
        self._display.reset_zoom()
