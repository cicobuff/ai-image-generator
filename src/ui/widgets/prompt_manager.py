"""Prompt manager widget for managing prompt lists and words."""

import random
from pathlib import Path
from typing import Callable, Optional

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from src.core.config import config_manager


class PromptListItem(Gtk.Box):
    """A single prompt list item with checkbox and count."""

    def __init__(
        self,
        name: str,
        on_toggled: Optional[Callable[[str, bool], None]] = None,
        on_selected: Optional[Callable[[str], None]] = None,
        on_count_changed: Optional[Callable[[str, int], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._name = name
        self._on_toggled = on_toggled
        self._on_selected = on_selected
        self._on_count_changed = on_count_changed

        self._build_ui()

    def _build_ui(self):
        """Build the list item UI."""
        # Checkbox for enabling/disabling
        self._checkbox = Gtk.CheckButton()
        self._checkbox.connect("toggled", self._on_checkbox_toggled)
        self.append(self._checkbox)

        # Label wrapped in an event box for click handling
        label_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        label_box.set_hexpand(True)
        self.append(label_box)

        self._label = Gtk.Label(label=self._name)
        self._label.set_halign(Gtk.Align.START)
        self._label.set_hexpand(True)
        self._label.add_css_class("caption")
        label_box.append(self._label)

        # Make the label box clickable for selection
        click = Gtk.GestureClick()
        click.connect("pressed", self._on_clicked)
        label_box.add_controller(click)

        # Dropdown for count (1-10)
        self._count_dropdown = Gtk.DropDown()
        count_model = Gtk.StringList.new([str(i) for i in range(1, 11)])
        self._count_dropdown.set_model(count_model)
        self._count_dropdown.set_selected(0)  # Default to "1"
        self._count_dropdown.add_css_class("prompt-list-count")
        self._count_dropdown.connect("notify::selected", self._on_count_value_changed)
        self.append(self._count_dropdown)

    def _on_checkbox_toggled(self, checkbox: Gtk.CheckButton):
        """Handle checkbox toggle."""
        if self._on_toggled:
            self._on_toggled(self._name, checkbox.get_active())

    def _on_clicked(self, gesture, n_press, x, y):
        """Handle click on the row."""
        if self._on_selected:
            self._on_selected(self._name)

    def _on_count_value_changed(self, dropdown: Gtk.DropDown, param):
        """Handle count value change."""
        if self._on_count_changed:
            self._on_count_changed(self._name, self.count)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_checked(self) -> bool:
        return self._checkbox.get_active()

    @property
    def count(self) -> int:
        return self._count_dropdown.get_selected() + 1  # Index 0 = value 1

    def set_checked(self, checked: bool):
        """Set the checkbox state."""
        self._checkbox.set_active(checked)

    def set_count(self, count: int):
        """Set the count value (1-10)."""
        if 1 <= count <= 10:
            self._count_dropdown.set_selected(count - 1)  # Value 1 = index 0

    def set_selected(self, selected: bool):
        """Set the visual selection state."""
        if selected:
            self.add_css_class("prompt-list-item-selected")
        else:
            self.remove_css_class("prompt-list-item-selected")


class PromptManagerPanel(Gtk.Box):
    """Panel for managing prompt lists and words."""

    def __init__(self, on_words_changed: Optional[Callable[[], None]] = None):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._on_words_changed = on_words_changed
        self._prompt_lists: dict[str, list[str]] = {}  # name -> words
        self._list_items: dict[str, PromptListItem] = {}  # name -> item widget
        self._selected_list: Optional[str] = None
        self._word_labels: list[Gtk.Box] = []  # Track word rows for selection
        self._selected_word_index: int = -1
        self._paned_position_restored = False  # Track if position was restored from config

        self._build_ui()
        self._load_prompt_lists()

    def _build_ui(self):
        """Build the panel UI."""
        self.add_css_class("prompt-manager-panel")

        # Horizontal paned for resizable List | Words
        self._paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self._paned.set_hexpand(True)
        self._paned.set_vexpand(True)
        self.append(self._paned)

        # Left side: Prompt Lists
        lists_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        lists_box.set_size_request(80, -1)  # Minimum width
        self._paned.set_start_child(lists_box)
        self._paned.set_resize_start_child(True)
        self._paned.set_shrink_start_child(True)

        # Lists header with add/remove buttons
        lists_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        lists_box.append(lists_header)

        lists_label = Gtk.Label(label="List")
        lists_label.add_css_class("section-header")
        lists_label.add_css_class("caption")
        lists_label.set_halign(Gtk.Align.START)
        lists_label.set_hexpand(True)
        lists_label.set_margin_start(8)  # Add left padding
        lists_header.append(lists_label)

        # Add list button (icon with border)
        self._add_list_btn = Gtk.Button.new_from_icon_name("list-add-symbolic")
        self._add_list_btn.add_css_class("prompt-manager-btn")
        self._add_list_btn.set_tooltip_text("Add new prompt list")
        self._add_list_btn.connect("clicked", self._on_add_list_clicked)
        lists_header.append(self._add_list_btn)

        # Remove list button (icon with border)
        self._remove_list_btn = Gtk.Button.new_from_icon_name("list-remove-symbolic")
        self._remove_list_btn.add_css_class("prompt-manager-btn")
        self._remove_list_btn.set_tooltip_text("Delete selected prompt list")
        self._remove_list_btn.connect("clicked", self._on_delete_list_clicked)
        self._remove_list_btn.set_sensitive(False)
        lists_header.append(self._remove_list_btn)

        # Scrolled window for lists
        lists_scroll = Gtk.ScrolledWindow()
        lists_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        lists_scroll.set_vexpand(True)
        lists_box.append(lists_scroll)

        self._lists_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        lists_scroll.set_child(self._lists_container)

        # Right side: Words
        words_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        words_box.set_size_request(80, -1)  # Minimum width
        self._paned.set_end_child(words_box)
        self._paned.set_resize_end_child(True)
        self._paned.set_shrink_end_child(True)

        # Words header with add/remove buttons
        words_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        words_box.append(words_header)

        words_label = Gtk.Label(label="Words")
        words_label.add_css_class("section-header")
        words_label.add_css_class("caption")
        words_label.set_halign(Gtk.Align.START)
        words_label.set_hexpand(True)
        words_label.set_margin_start(8)  # Add left padding
        words_header.append(words_label)

        # Add word button (icon with border)
        self._add_word_btn = Gtk.Button.new_from_icon_name("list-add-symbolic")
        self._add_word_btn.add_css_class("prompt-manager-btn")
        self._add_word_btn.set_tooltip_text("Add word to selected list")
        self._add_word_btn.connect("clicked", self._on_add_word_clicked)
        self._add_word_btn.set_sensitive(False)
        words_header.append(self._add_word_btn)

        # Remove word button (icon with border)
        self._remove_word_btn = Gtk.Button.new_from_icon_name("list-remove-symbolic")
        self._remove_word_btn.add_css_class("prompt-manager-btn")
        self._remove_word_btn.set_tooltip_text("Remove selected word")
        self._remove_word_btn.connect("clicked", self._on_delete_word_clicked)
        self._remove_word_btn.set_sensitive(False)
        words_header.append(self._remove_word_btn)

        # Scrolled window for words
        words_scroll = Gtk.ScrolledWindow()
        words_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        words_scroll.set_vexpand(True)
        words_box.append(words_scroll)

        self._words_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        words_scroll.set_child(self._words_container)

        # Set initial position to split evenly
        self._paned.connect("realize", self._on_paned_realize)

    def _on_paned_realize(self, widget):
        """Set equal sizes when paned is realized."""
        # Skip if position was restored from config
        if self._paned_position_restored:
            return
        width = self._paned.get_allocated_width()
        if width > 0:
            self._paned.set_position(width // 2)

    def _get_prompt_lists_dir(self) -> Path:
        """Get the prompt lists directory path."""
        return config_manager.config.get_models_path() / "prompt-lists"

    def _load_prompt_lists(self):
        """Load all .txt files from models/prompt-lists/."""
        lists_dir = self._get_prompt_lists_dir()
        if not lists_dir.exists():
            lists_dir.mkdir(parents=True, exist_ok=True)
            return

        # Clear existing items
        self._prompt_lists.clear()
        self._list_items.clear()
        while self._lists_container.get_first_child():
            self._lists_container.remove(self._lists_container.get_first_child())

        # Load each .txt file
        for txt_file in sorted(lists_dir.glob("*.txt")):
            name = txt_file.stem
            words = self._load_words_from_file(txt_file)
            self._prompt_lists[name] = words
            self._add_list_item(name)

    def _load_words_from_file(self, file_path: Path) -> list[str]:
        """Load words from a file, one per line."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def _save_prompt_list(self, name: str, words: list[str]):
        """Save words to models/prompt-lists/{name}.txt."""
        lists_dir = self._get_prompt_lists_dir()
        lists_dir.mkdir(parents=True, exist_ok=True)

        file_path = lists_dir / f"{name}.txt"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for word in words:
                    f.write(f"{word}\n")
        except Exception as e:
            print(f"Error saving {file_path}: {e}")

    def _delete_prompt_list_file(self, name: str):
        """Delete a prompt list file."""
        lists_dir = self._get_prompt_lists_dir()
        file_path = lists_dir / f"{name}.txt"
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    def _add_list_item(self, name: str):
        """Add a list item widget."""
        item = PromptListItem(
            name=name,
            on_toggled=self._on_list_toggled,
            on_selected=self._on_list_selected,
        )
        item.add_css_class("prompt-list-item")
        self._lists_container.append(item)
        self._list_items[name] = item

    def _on_list_toggled(self, name: str, checked: bool):
        """Handle list checkbox toggle."""
        if self._on_words_changed:
            self._on_words_changed()

    def _on_list_selected(self, name: str):
        """Handle list selection."""
        # Deselect previous
        if self._selected_list and self._selected_list in self._list_items:
            self._list_items[self._selected_list].set_selected(False)

        # Select new
        self._selected_list = name
        if name in self._list_items:
            self._list_items[name].set_selected(True)

        # Update buttons
        self._remove_list_btn.set_sensitive(True)
        self._add_word_btn.set_sensitive(True)

        # Update words display
        self._update_words_display()

    def _update_words_display(self):
        """Update the words display for the selected list."""
        # Clear existing words
        self._word_labels.clear()
        self._selected_word_index = -1
        self._remove_word_btn.set_sensitive(False)
        while self._words_container.get_first_child():
            self._words_container.remove(self._words_container.get_first_child())

        if not self._selected_list or self._selected_list not in self._prompt_lists:
            return

        words = self._prompt_lists[self._selected_list]
        for i, word in enumerate(words):
            row = self._create_word_row(word, i)
            self._words_container.append(row)
            self._word_labels.append(row)

    def _create_word_row(self, word: str, index: int) -> Gtk.Box:
        """Create a word row widget."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        row.add_css_class("prompt-word-row")

        label = Gtk.Label(label=word)
        label.set_halign(Gtk.Align.START)
        label.set_hexpand(True)
        label.add_css_class("caption")
        row.append(label)

        # Make clickable for selection
        click = Gtk.GestureClick()
        click.connect("pressed", lambda g, n, x, y, i=index: self._on_word_clicked(i))
        row.add_controller(click)

        return row

    def _on_word_clicked(self, index: int):
        """Handle word selection."""
        # Deselect previous
        if 0 <= self._selected_word_index < len(self._word_labels):
            self._word_labels[self._selected_word_index].remove_css_class("prompt-word-selected")

        # Select new
        self._selected_word_index = index
        if 0 <= index < len(self._word_labels):
            self._word_labels[index].add_css_class("prompt-word-selected")
            self._remove_word_btn.set_sensitive(True)

    def _on_add_list_clicked(self, button: Gtk.Button):
        """Show dialog to create new prompt list."""
        dialog = Gtk.Dialog(title="New Prompt List")
        dialog.set_modal(True)
        dialog.set_default_size(300, -1)

        # Get the root window
        root = self.get_root()
        if root:
            dialog.set_transient_for(root)

        # Content area
        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(12)

        label = Gtk.Label(label="Enter name for the new prompt list:")
        label.set_halign(Gtk.Align.START)
        content.append(label)

        entry = Gtk.Entry()
        entry.set_placeholder_text("e.g., quality, style, faces")
        content.append(entry)

        # Buttons
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        create_btn = dialog.add_button("Create", Gtk.ResponseType.OK)
        create_btn.add_css_class("suggested-action")

        def on_response(dialog, response):
            if response == Gtk.ResponseType.OK:
                name = entry.get_text().strip()
                if name and self._validate_list_name(name):
                    self._create_new_list(name)
            dialog.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _validate_list_name(self, name: str) -> bool:
        """Validate a list name (no special characters, not empty)."""
        if not name:
            return False
        # Only allow alphanumeric, dash, underscore
        for c in name:
            if not (c.isalnum() or c in "-_"):
                return False
        # Check if already exists
        if name in self._prompt_lists:
            return False
        return True

    def _create_new_list(self, name: str):
        """Create a new empty prompt list."""
        self._prompt_lists[name] = []
        self._add_list_item(name)
        self._save_prompt_list(name, [])
        # Select the new list
        self._on_list_selected(name)

    def _on_delete_list_clicked(self, button: Gtk.Button):
        """Delete selected prompt list."""
        if not self._selected_list:
            return

        name = self._selected_list

        # Remove from UI
        if name in self._list_items:
            self._lists_container.remove(self._list_items[name])
            del self._list_items[name]

        # Remove from data
        if name in self._prompt_lists:
            del self._prompt_lists[name]

        # Delete file
        self._delete_prompt_list_file(name)

        # Clear selection
        self._selected_list = None
        self._remove_list_btn.set_sensitive(False)
        self._add_word_btn.set_sensitive(False)
        self._update_words_display()

        if self._on_words_changed:
            self._on_words_changed()

    def _on_add_word_clicked(self, button: Gtk.Button):
        """Add word to selected list."""
        if not self._selected_list:
            return

        dialog = Gtk.Dialog(title="Add Word")
        dialog.set_modal(True)
        dialog.set_default_size(300, -1)

        root = self.get_root()
        if root:
            dialog.set_transient_for(root)

        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(12)

        label = Gtk.Label(label="Enter word or phrase:")
        label.set_halign(Gtk.Align.START)
        content.append(label)

        entry = Gtk.Entry()
        entry.set_placeholder_text("e.g., masterpiece, best quality")
        content.append(entry)

        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        add_btn = dialog.add_button("Add", Gtk.ResponseType.OK)
        add_btn.add_css_class("suggested-action")

        def on_response(dialog, response):
            if response == Gtk.ResponseType.OK:
                word = entry.get_text().strip()
                if word:
                    self._add_word_to_list(word)
            dialog.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _add_word_to_list(self, word: str):
        """Add a word to the selected list."""
        if not self._selected_list:
            return

        words = self._prompt_lists.get(self._selected_list, [])
        if word not in words:  # Avoid duplicates
            words.append(word)
            self._prompt_lists[self._selected_list] = words
            self._save_prompt_list(self._selected_list, words)
            self._update_words_display()

            if self._on_words_changed:
                self._on_words_changed()

    def _on_delete_word_clicked(self, button: Gtk.Button):
        """Remove selected word from list."""
        if not self._selected_list or self._selected_word_index < 0:
            return

        words = self._prompt_lists.get(self._selected_list, [])
        if 0 <= self._selected_word_index < len(words):
            words.pop(self._selected_word_index)
            self._prompt_lists[self._selected_list] = words
            self._save_prompt_list(self._selected_list, words)
            self._update_words_display()

            if self._on_words_changed:
                self._on_words_changed()

    def get_checked_words(self) -> list[str]:
        """Get randomly selected words from checked prompt lists.

        For each checked list, randomly selects N words where N is the count
        value from the dropdown. Each word can only be picked once per list.
        If count > number of words, all words are selected.
        """
        selected_words = []
        for name, item in self._list_items.items():
            if item.is_checked and name in self._prompt_lists:
                list_words = self._prompt_lists[name]
                count = item.count

                if not list_words:
                    continue

                # If count >= number of words, select all
                if count >= len(list_words):
                    selected_words.extend(list_words)
                else:
                    # Randomly select 'count' words without replacement
                    selected = random.sample(list_words, count)
                    selected_words.extend(selected)

        return selected_words

    def get_checked_words_string(self) -> str:
        """Get randomly selected words from checked lists as a comma-separated string."""
        words = self.get_checked_words()
        return ", ".join(words) if words else ""

    def get_paned_position(self) -> int:
        """Get the current paned position (List/Words split)."""
        return self._paned.get_position()

    def set_paned_position(self, position: int):
        """Set the paned position (List/Words split)."""
        if position > 0:
            self._paned.set_position(position)
            self._paned_position_restored = True
