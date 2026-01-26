"""Thumbnail gallery widget for displaying generated images with lazy loading."""

from pathlib import Path
from typing import Optional, Callable, List, Set
from enum import Enum
import io
import threading
import queue

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

from PIL import Image

from src.utils.constants import THUMBNAIL_SIZE, THUMBNAIL_COLUMNS
from src.ui.widgets.info_helper import SectionHeader, SECTION_INFO


class SortOrder(Enum):
    """Sort order for thumbnails."""
    DATE_DESC = "date_desc"  # Newest first
    DATE_ASC = "date_asc"    # Oldest first


class ThumbnailItem(Gtk.Button):
    """Single thumbnail item in the gallery with lazy loading support."""

    def __init__(
        self,
        path: Path,
        thumbnail_size: int = THUMBNAIL_SIZE,
        image: Optional[Image.Image] = None,
        on_click: Optional[Callable[[Path], None]] = None,
        placeholder: bool = False,
    ):
        super().__init__()
        self._path = path
        self._on_click = on_click
        self._thumbnail_size = thumbnail_size
        self._original_image = image
        self._is_loaded = False
        self._pixbuf: Optional[GdkPixbuf.Pixbuf] = None

        self.add_css_class("thumbnail")
        self.set_has_frame(False)

        if placeholder:
            self._build_placeholder()
        else:
            self._build_ui(image)

        self.connect("clicked", self._on_clicked)

    def _build_placeholder(self):
        """Build a placeholder thumbnail (blank/loading state)."""
        # Create a simple placeholder box
        placeholder = Gtk.Box()
        placeholder.set_size_request(self._thumbnail_size, self._thumbnail_size)
        placeholder.add_css_class("thumbnail-placeholder")
        self.set_child(placeholder)
        self._is_loaded = False

    def _build_ui(self, image: Optional[Image.Image]):
        """Build the thumbnail UI with actual image."""
        # Create thumbnail image
        if image:
            self._pixbuf = self._pil_to_thumbnail(image)
        else:
            # Load from path
            try:
                self._pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    str(self._path),
                    self._thumbnail_size,
                    self._thumbnail_size,
                    True,  # Preserve aspect ratio
                )
            except Exception:
                self._pixbuf = None

        if self._pixbuf:
            picture = Gtk.Picture.new_for_pixbuf(self._pixbuf)
            picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            picture.set_size_request(self._thumbnail_size, self._thumbnail_size)
            self.set_child(picture)
            self._is_loaded = True
        else:
            # Error placeholder
            label = Gtk.Label(label="?")
            label.set_size_request(self._thumbnail_size, self._thumbnail_size)
            self.set_child(label)
            self._is_loaded = True

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

    def load_thumbnail(self) -> bool:
        """Load the actual thumbnail image. Returns True if loading was needed."""
        if self._is_loaded:
            return False

        self._build_ui(self._original_image)
        return True

    def load_thumbnail_from_pixbuf(self, pixbuf: GdkPixbuf.Pixbuf):
        """Set the thumbnail from a pre-loaded pixbuf."""
        self._pixbuf = pixbuf
        picture = Gtk.Picture.new_for_pixbuf(pixbuf)
        picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        picture.set_size_request(self._thumbnail_size, self._thumbnail_size)
        self.set_child(picture)
        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        """Check if the thumbnail is loaded."""
        return self._is_loaded

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
        self._is_loaded = False
        # Rebuild as placeholder, will be reloaded
        self._build_placeholder()


class ThumbnailGallery(Gtk.Box):
    """Scrollable gallery of image thumbnails with lazy loading."""

    MIN_THUMBNAIL_SIZE = 64
    MAX_THUMBNAIL_SIZE = 256
    DEFAULT_THUMBNAIL_SIZE = THUMBNAIL_SIZE
    INITIAL_VISIBLE_COUNT = 20  # Number of thumbnails to load immediately
    BATCH_LOAD_SIZE = 10  # Number of thumbnails to load per batch in background

    def __init__(
        self,
        on_image_selected: Optional[Callable[[Path], None]] = None,
        on_directory_changed: Optional[Callable[[Path], None]] = None,
        on_image_deleted: Optional[Callable[[Path], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._on_image_selected = on_image_selected
        self._on_directory_changed = on_directory_changed
        self._on_image_deleted = on_image_deleted
        self._thumbnails: List[ThumbnailItem] = []
        self._image_paths: List[Path] = []  # Store paths for re-sorting
        self._selected_path: Optional[Path] = None
        self._thumbnail_size = self.DEFAULT_THUMBNAIL_SIZE
        self._sort_order = SortOrder.DATE_DESC
        self._base_directory: Optional[Path] = None  # Root output directory
        self._current_directory: Optional[Path] = None  # Currently selected subdirectory
        self._subdirectories: List[str] = []  # List of subdirectory names

        # Background loading state
        self._load_queue: queue.Queue = queue.Queue()
        self._loading_thread: Optional[threading.Thread] = None
        self._stop_loading = threading.Event()
        self._loaded_indices: Set[int] = set()

        self.add_css_class("thumbnail-gallery")
        self._build_ui()
        self._setup_key_controller()

    def _build_ui(self):
        """Build the gallery UI."""
        # Header with info button
        header = SectionHeader("Gallery", SECTION_INFO["gallery"])
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
        self._scrolled = Gtk.ScrolledWindow()
        self._scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._scrolled.set_vexpand(True)
        self._scrolled.set_margin_top(4)
        self.append(self._scrolled)

        # Flow box for thumbnails
        self._flowbox = Gtk.FlowBox()
        self._flowbox.set_valign(Gtk.Align.START)
        self._flowbox.set_max_children_per_line(10)
        self._flowbox.set_min_children_per_line(1)
        self._flowbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self._flowbox.set_homogeneous(True)
        self._flowbox.set_row_spacing(4)
        self._flowbox.set_column_spacing(4)
        self._scrolled.set_child(self._flowbox)

    def _setup_key_controller(self):
        """Set up keyboard controller for Delete key."""
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)
        # Make widget focusable
        self.set_focusable(True)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events."""
        if keyval == Gdk.KEY_Delete or keyval == Gdk.KEY_BackSpace:
            self.delete_selected()
            return True

        # Arrow key navigation
        if keyval in (Gdk.KEY_Left, Gdk.KEY_Right, Gdk.KEY_Up, Gdk.KEY_Down):
            return self._handle_arrow_navigation(keyval)

        return False

    def _handle_arrow_navigation(self, keyval) -> bool:
        """Handle arrow key navigation between thumbnails."""
        if not self._thumbnails:
            return False

        # Get current selection index
        current_index = self._get_selected_index()
        if current_index < 0:
            # No selection, select first item
            if self._thumbnails:
                self._select_thumbnail_at_index(0)
            return True

        # Calculate new index based on arrow key
        new_index = current_index

        if keyval == Gdk.KEY_Left:
            new_index = current_index - 1
        elif keyval == Gdk.KEY_Right:
            new_index = current_index + 1
        elif keyval == Gdk.KEY_Up or keyval == Gdk.KEY_Down:
            # Calculate columns per row for up/down navigation
            columns = self._get_columns_per_row()
            if keyval == Gdk.KEY_Up:
                new_index = current_index - columns
            else:
                new_index = current_index + columns

        # Clamp to valid range
        if new_index < 0 or new_index >= len(self._thumbnails):
            return True  # Handled but no change

        self._select_thumbnail_at_index(new_index)
        return True

    def _get_selected_index(self) -> int:
        """Get the index of the currently selected thumbnail."""
        if self._selected_path is None:
            return -1
        for i, thumb in enumerate(self._thumbnails):
            if thumb.path == self._selected_path:
                return i
        return -1

    def _get_columns_per_row(self) -> int:
        """Calculate how many columns are visible per row based on flowbox width."""
        # Get flowbox allocated width
        width = self._flowbox.get_allocated_width()
        if width <= 0:
            return 1

        # Calculate columns: (width) / (thumbnail_size + column_spacing)
        item_width = self._thumbnail_size + 4  # 4 is column spacing
        columns = max(1, width // item_width)

        # Respect max children per line setting
        max_columns = self._flowbox.get_max_children_per_line()
        return min(columns, max_columns)

    def _select_thumbnail_at_index(self, index: int):
        """Select thumbnail at given index and scroll it into view."""
        if index < 0 or index >= len(self._thumbnails):
            return

        thumb = self._thumbnails[index]
        path = thumb.path

        # Update selection state
        self._selected_path = path
        for t in self._thumbnails:
            t.set_selected(t.path == path)

        # Scroll thumbnail into view
        self._scroll_thumbnail_into_view(thumb)

        # Trigger callback to load image in center display
        if self._on_image_selected:
            self._on_image_selected(path)

    def _scroll_thumbnail_into_view(self, thumbnail: ThumbnailItem):
        """Scroll the gallery to ensure the thumbnail is visible."""
        # Get the FlowBoxChild that contains the thumbnail
        flowbox_child = thumbnail.get_parent()
        if flowbox_child is None:
            return

        # Get the vertical adjustment
        vadj = self._scrolled.get_vadjustment()
        if vadj is None:
            return

        # Get the child's allocation (position within flowbox)
        allocation = flowbox_child.get_allocation()

        # Get current scroll position and visible area
        scroll_top = vadj.get_value()
        visible_height = vadj.get_page_size()
        scroll_bottom = scroll_top + visible_height

        # Calculate thumbnail bounds
        thumb_top = allocation.y
        thumb_bottom = allocation.y + allocation.height

        # Scroll if needed
        if thumb_top < scroll_top:
            # Thumbnail is above visible area
            vadj.set_value(thumb_top)
        elif thumb_bottom > scroll_bottom:
            # Thumbnail is below visible area
            vadj.set_value(thumb_bottom - visible_height)

    def _stop_background_loading(self):
        """Stop any running background loading."""
        self._stop_loading.set()
        if self._loading_thread is not None and self._loading_thread.is_alive():
            self._loading_thread.join(timeout=0.5)
        self._loading_thread = None
        # Clear the queue
        while not self._load_queue.empty():
            try:
                self._load_queue.get_nowait()
            except queue.Empty:
                break

    def _start_background_loading(self, start_index: int):
        """Start background loading from the given index."""
        self._stop_loading.clear()

        # Queue up indices to load
        for i in range(start_index, len(self._thumbnails)):
            if i not in self._loaded_indices:
                self._load_queue.put(i)

        # Start loading thread
        self._loading_thread = threading.Thread(
            target=self._background_load_worker,
            daemon=True
        )
        self._loading_thread.start()

    def _background_load_worker(self):
        """Background thread worker for loading thumbnails."""
        while not self._stop_loading.is_set():
            try:
                index = self._load_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._stop_loading.is_set():
                break

            if index >= len(self._thumbnails) or index in self._loaded_indices:
                continue

            # Load the thumbnail in background
            try:
                thumb = self._thumbnails[index]
                path = thumb.path

                # Load and scale the image
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    str(path),
                    self._thumbnail_size,
                    self._thumbnail_size,
                    True,
                )

                if self._stop_loading.is_set():
                    break

                # Schedule UI update on main thread
                GLib.idle_add(self._update_thumbnail_ui, index, pixbuf)
                self._loaded_indices.add(index)

            except Exception as e:
                # Mark as loaded even on error to prevent retry loops
                self._loaded_indices.add(index)

        # Clear remaining queue items
        while not self._load_queue.empty():
            try:
                self._load_queue.get_nowait()
            except queue.Empty:
                break

    def _update_thumbnail_ui(self, index: int, pixbuf: GdkPixbuf.Pixbuf) -> bool:
        """Update thumbnail UI on main thread. Returns False to remove from idle."""
        if index < len(self._thumbnails):
            thumb = self._thumbnails[index]
            if not thumb.is_loaded:
                thumb.load_thumbnail_from_pixbuf(pixbuf)
        return False

    def delete_selected(self) -> bool:
        """Delete the currently selected image."""
        if self._selected_path is None:
            return False

        path_to_delete = self._selected_path

        # Find the thumbnail widget
        thumbnail_to_remove = None
        thumbnail_index = -1
        for i, thumb in enumerate(self._thumbnails):
            if thumb.path == path_to_delete:
                thumbnail_to_remove = thumb
                thumbnail_index = i
                break

        if thumbnail_to_remove is None:
            return False

        try:
            # Delete the file from disk
            if path_to_delete.exists():
                path_to_delete.unlink()

            # Remove from internal lists
            if path_to_delete in self._image_paths:
                self._image_paths.remove(path_to_delete)

            # Remove from loaded indices (shift all indices after this one)
            self._loaded_indices.discard(thumbnail_index)
            new_loaded = set()
            for idx in self._loaded_indices:
                if idx > thumbnail_index:
                    new_loaded.add(idx - 1)
                else:
                    new_loaded.add(idx)
            self._loaded_indices = new_loaded

            # Remove thumbnail widget
            self._flowbox.remove(thumbnail_to_remove)
            self._thumbnails.remove(thumbnail_to_remove)

            # Select next image if available
            self._selected_path = None
            if self._thumbnails:
                next_index = min(thumbnail_index, len(self._thumbnails) - 1)
                if next_index >= 0:
                    next_thumb = self._thumbnails[next_index]
                    next_thumb.set_selected(True)
                    self._selected_path = next_thumb.path

            # Notify callback
            if self._on_image_deleted:
                self._on_image_deleted(path_to_delete)

            return True

        except Exception as e:
            print(f"Error deleting image {path_to_delete}: {e}")
            return False

    def _scan_subdirectories(self):
        """Scan for subdirectories in the base directory."""
        self._subdirectories = []
        if self._base_directory is None or not self._base_directory.exists():
            return

        for item in self._base_directory.iterdir():
            if item.is_dir():
                self._subdirectories.append(item.name)

        self._subdirectories.sort()

    def _update_directory_combo(self):
        """Update the directory combo box."""
        self._dir_combo.handler_block_by_func(self._on_directory_combo_changed)

        self._dir_combo.remove_all()
        self._dir_combo.append_text("(root)")

        for subdir in self._subdirectories:
            self._dir_combo.append_text(subdir)

        if self._current_directory == self._base_directory:
            self._dir_combo.set_active(0)
        else:
            rel_path = self._get_relative_subdir()
            if rel_path in self._subdirectories:
                idx = self._subdirectories.index(rel_path) + 1
                self._dir_combo.set_active(idx)
            else:
                entry = self._dir_combo.get_child()
                if entry and rel_path:
                    entry.set_text(rel_path)

        self._dir_combo.handler_unblock_by_func(self._on_directory_combo_changed)

    def _get_relative_subdir(self) -> str:
        """Get the relative subdirectory name."""
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

        active_idx = combo.get_active()
        if active_idx == 0:
            new_dir = self._base_directory
        elif active_idx > 0:
            subdir_name = self._subdirectories[active_idx - 1]
            new_dir = self._base_directory / subdir_name
        else:
            entry = combo.get_child()
            if entry:
                text = entry.get_text().strip()
                if text and text != "(root)":
                    new_dir = self._base_directory / text
                else:
                    new_dir = self._base_directory
            else:
                new_dir = self._base_directory

        if new_dir != self._current_directory:
            self._current_directory = new_dir

            if self._on_directory_changed:
                self._on_directory_changed(new_dir)

            if new_dir.exists():
                self._load_thumbnails_from_current_directory()
            else:
                self._clear_thumbnails()

    def _on_refresh_directories(self, button):
        """Handle refresh directories button click."""
        self._scan_subdirectories()
        self._update_directory_combo()

    def _on_sort_clicked(self, button):
        """Handle sort button click."""
        if self._sort_order == SortOrder.DATE_DESC:
            self._sort_order = SortOrder.DATE_ASC
            self._sort_button.set_icon_name("view-sort-ascending-symbolic")
            self._sort_button.set_tooltip_text("Sort: Oldest first (click to change)")
        else:
            self._sort_order = SortOrder.DATE_DESC
            self._sort_button.set_icon_name("view-sort-descending-symbolic")
            self._sort_button.set_tooltip_text("Sort: Newest first (click to change)")

        self._refresh_thumbnails()

    def _on_size_changed(self, scale):
        """Handle thumbnail size change."""
        new_size = int(scale.get_value())
        if new_size == self._thumbnail_size:
            return

        self._thumbnail_size = new_size
        self._size_scale.set_tooltip_text(f"Thumbnail size: {new_size}px")

        self._refresh_thumbnails()

    def _clear_thumbnails(self):
        """Clear all thumbnail widgets."""
        self._stop_background_loading()
        for thumb in self._thumbnails:
            self._flowbox.remove(thumb)
        self._thumbnails.clear()
        self._image_paths.clear()
        self._selected_path = None
        self._loaded_indices.clear()

    def _refresh_thumbnails(self):
        """Refresh all thumbnails with current settings."""
        if not self._image_paths:
            return

        # Stop any background loading
        self._stop_background_loading()

        # Store selected path
        selected = self._selected_path

        # Clear current thumbnails
        for thumb in self._thumbnails:
            self._flowbox.remove(thumb)
        self._thumbnails.clear()
        self._loaded_indices.clear()

        # Sort paths
        sorted_paths = self._sort_paths(self._image_paths)

        # Determine how many to load immediately
        initial_count = min(self.INITIAL_VISIBLE_COUNT, len(sorted_paths))

        # Create all thumbnails - first batch loaded, rest as placeholders
        for i, path in enumerate(sorted_paths):
            if i < initial_count:
                # Load immediately
                thumbnail = ThumbnailItem(
                    path=path,
                    thumbnail_size=self._thumbnail_size,
                    on_click=self._on_thumbnail_clicked,
                    placeholder=False,
                )
                self._loaded_indices.add(i)
            else:
                # Create as placeholder
                thumbnail = ThumbnailItem(
                    path=path,
                    thumbnail_size=self._thumbnail_size,
                    on_click=self._on_thumbnail_clicked,
                    placeholder=True,
                )

            self._thumbnails.append(thumbnail)
            self._flowbox.append(thumbnail)

        # Restore selection
        if selected:
            for thumb in self._thumbnails:
                thumb.set_selected(thumb.path == selected)

        # Start background loading for the rest
        if initial_count < len(sorted_paths):
            self._start_background_loading(initial_count)

    def _sort_paths(self, paths: List[Path]) -> List[Path]:
        """Sort paths according to current sort order."""
        reverse = self._sort_order == SortOrder.DATE_DESC
        return sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=reverse)

    def _load_thumbnails_from_current_directory(self):
        """Load thumbnails from the current directory with lazy loading."""
        self._clear_thumbnails()

        if self._current_directory is None or not self._current_directory.exists():
            return

        # Get image files (only in current directory)
        image_files = []
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            for item in self._current_directory.iterdir():
                if item.is_file() and item.suffix.lower() == ext:
                    image_files.append(item)

        # Store paths
        self._image_paths = image_files

        # Sort according to current order
        sorted_files = self._sort_paths(image_files)

        # Determine how many to load immediately
        initial_count = min(self.INITIAL_VISIBLE_COUNT, len(sorted_files))

        # Create thumbnails - first batch loaded, rest as placeholders
        for i, path in enumerate(sorted_files):
            if i < initial_count:
                # Load immediately for visible area
                thumbnail = ThumbnailItem(
                    path=path,
                    thumbnail_size=self._thumbnail_size,
                    on_click=self._on_thumbnail_clicked,
                    placeholder=False,
                )
                self._loaded_indices.add(i)
            else:
                # Create placeholder for background loading
                thumbnail = ThumbnailItem(
                    path=path,
                    thumbnail_size=self._thumbnail_size,
                    on_click=self._on_thumbnail_clicked,
                    placeholder=True,
                )

            self._thumbnails.append(thumbnail)
            self._flowbox.append(thumbnail)

        # Start background loading for remaining thumbnails
        if initial_count < len(sorted_files):
            self._start_background_loading(initial_count)

    def add_image(self, path: Path, image: Optional[Image.Image] = None):
        """Add a new image to the gallery."""
        if self._current_directory and path.parent == self._current_directory:
            if path not in self._image_paths:
                self._image_paths.append(path)

            # New images are always loaded immediately (not placeholders)
            thumbnail = ThumbnailItem(
                path=path,
                thumbnail_size=self._thumbnail_size,
                image=image,
                on_click=self._on_thumbnail_clicked,
                placeholder=False,
            )

            if self._sort_order == SortOrder.DATE_DESC:
                self._thumbnails.insert(0, thumbnail)
                self._flowbox.prepend(thumbnail)
                # Shift loaded indices
                new_loaded = {0}
                for idx in self._loaded_indices:
                    new_loaded.add(idx + 1)
                self._loaded_indices = new_loaded
            else:
                self._thumbnails.append(thumbnail)
                self._flowbox.append(thumbnail)
                self._loaded_indices.add(len(self._thumbnails) - 1)

    def _on_thumbnail_clicked(self, path: Path):
        """Handle thumbnail click."""
        self._selected_path = path
        for thumb in self._thumbnails:
            thumb.set_selected(thumb.path == path)

        # Grab focus so arrow keys work for navigation
        self.grab_focus()

        if self._on_image_selected:
            self._on_image_selected(path)

    def clear(self):
        """Clear all thumbnails."""
        self._clear_thumbnails()

    def load_from_directory(self, directory: Path):
        """Set the base directory and load thumbnails."""
        self._base_directory = directory
        self._current_directory = directory

        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

        self._scan_subdirectories()
        self._update_directory_combo()
        self._load_thumbnails_from_current_directory()

    def get_selected_path(self) -> Optional[Path]:
        """Get the currently selected image path."""
        return self._selected_path

    def select_path(self, path: Path):
        """Select a specific image by path."""
        self._on_thumbnail_clicked(path)

    def get_output_directory(self) -> Path:
        """Get the current output directory."""
        if self._current_directory is not None:
            return self._current_directory
        if self._base_directory is not None:
            return self._base_directory
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

        entry = self._dir_combo.get_child()
        if entry:
            if subdir_name and subdir_name != "(root)":
                entry.set_text(subdir_name)
            else:
                self._dir_combo.set_active(0)

        if self._on_directory_changed:
            self._on_directory_changed(new_dir)

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
