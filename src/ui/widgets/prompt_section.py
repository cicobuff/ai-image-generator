"""Prompt section widget containing prompt manager and prompt entries."""

from typing import Optional

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from src.ui.widgets.prompt_manager import PromptManagerPanel
from src.ui.widgets.prompt_entry import PromptEntry
from src.ui.widgets.info_helper import SectionHeader, SECTION_INFO, add_hover_tooltip, LABEL_TOOLTIPS


class PromptSection(Gtk.Box):
    """Section containing prompt manager and prompt entries with resizable panes."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("prompt-section")
        self._prompts_split_restored = False  # Track if position was restored from config

        self._build_ui()

    def _build_ui(self):
        """Build the section UI."""
        # Section header with info button
        header = SectionHeader("Prompt Management", SECTION_INFO.get("prompt_management"))
        header.set_margin_bottom(8)
        self.append(header)

        # Main horizontal paned: Prompts | Manager
        self._main_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self._main_paned.set_vexpand(True)
        self.append(self._main_paned)

        # Left side: Prompts with vertical paned
        prompts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        prompts_box.set_margin_end(8)
        self._main_paned.set_start_child(prompts_box)
        self._main_paned.set_resize_start_child(True)
        self._main_paned.set_shrink_start_child(True)

        # "Prompts" label for left side with tooltip
        prompts_label = Gtk.Label(label="Prompts")
        prompts_label.add_css_class("section-header")
        prompts_label.add_css_class("caption")
        prompts_label.set_halign(Gtk.Align.START)
        add_hover_tooltip(prompts_label, "Enter your positive and negative prompts here. Random words from checked lists are added to positive prompt.")
        prompts_box.append(prompts_label)

        # Vertical paned for positive/negative prompts
        self._prompts_paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self._prompts_paned.set_vexpand(True)
        self._prompts_paned.connect("realize", self._on_prompts_paned_realize)
        prompts_box.append(self._prompts_paned)

        # Top section: horizontal box for positive prompt and refiner prompt
        self._top_prompts_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._top_prompts_box.set_vexpand(True)
        self._prompts_paned.set_start_child(self._top_prompts_box)
        self._prompts_paned.set_resize_start_child(True)
        self._prompts_paned.set_shrink_start_child(True)

        # Positive prompt (left side of top)
        self._positive_entry = PromptEntry(
            label="Positive Prompt",
            is_positive=True,
            placeholder="Describe what you want to generate...",
        )
        self._positive_entry.set_size_request(-1, 60)  # Minimum height
        self._positive_entry.set_vexpand(True)
        self._positive_entry.set_hexpand(True)
        self._top_prompts_box.append(self._positive_entry)

        # Refiner prompt (right side of top, hidden by default)
        self._refiner_entry = PromptEntry(
            label="Refiner Prompt",
            style_type="refiner",  # Use yellow/amber border for refiner
            placeholder="Prompt for refining selected regions...",
        )
        self._refiner_entry.set_size_request(-1, 60)  # Minimum height
        self._refiner_entry.set_vexpand(True)
        self._refiner_entry.set_hexpand(True)
        self._refiner_entry.set_visible(False)  # Hidden by default
        self._top_prompts_box.append(self._refiner_entry)

        # Negative prompt (bottom)
        self._negative_entry = PromptEntry(
            label="Negative Prompt",
            is_positive=False,
            placeholder="Describe what you want to avoid...",
        )
        self._negative_entry.set_size_request(-1, 60)  # Minimum height
        self._negative_entry.set_vexpand(True)
        self._prompts_paned.set_end_child(self._negative_entry)
        self._prompts_paned.set_resize_end_child(True)
        self._prompts_paned.set_shrink_end_child(True)

        # Right side: Prompt Manager Panel
        self._prompt_manager = PromptManagerPanel(
            on_words_changed=self._on_prompt_words_changed,
            on_word_double_clicked=self._on_word_double_clicked,
        )
        self._prompt_manager.set_size_request(200, -1)  # Minimum width
        self._main_paned.set_end_child(self._prompt_manager)
        self._main_paned.set_resize_end_child(False)
        self._main_paned.set_shrink_end_child(True)

    def _on_prompt_words_changed(self):
        """Handle changes to prompt manager words."""
        # This could be used to auto-update prompts if desired
        pass

    def _on_word_double_clicked(self, word: str):
        """Handle double-click on a word - add it to positive prompt."""
        current_text = self._positive_entry.get_text().strip()

        if not current_text:
            # Empty prompt, just add the word
            self._positive_entry.set_text(word)
        elif current_text.endswith(","):
            # Already ends with comma, add space and word
            self._positive_entry.set_text(f"{current_text} {word}")
        else:
            # Add comma, space, then word
            self._positive_entry.set_text(f"{current_text}, {word}")

    def _on_prompts_paned_realize(self, widget):
        """Set equal sizes for prompts when paned is realized."""
        # Skip if position was restored from config
        if self._prompts_split_restored:
            return
        # Get allocated height and set position to half
        height = self._prompts_paned.get_allocated_height()
        if height > 0:
            self._prompts_paned.set_position(height // 2)

    @property
    def positive_entry(self) -> PromptEntry:
        """Get the positive prompt entry."""
        return self._positive_entry

    @property
    def negative_entry(self) -> PromptEntry:
        """Get the negative prompt entry."""
        return self._negative_entry

    @property
    def prompt_manager(self) -> PromptManagerPanel:
        """Get the prompt manager panel."""
        return self._prompt_manager

    def get_positive_prompt(self) -> str:
        """Get the positive prompt text, prepending checked words from manager."""
        user_prompt = self._positive_entry.get_text()
        checked_words = self._prompt_manager.get_checked_words_string()

        if checked_words and user_prompt:
            return f"{checked_words}, {user_prompt}"
        elif checked_words:
            return checked_words
        else:
            return user_prompt

    def get_negative_prompt(self) -> str:
        """Get the negative prompt text."""
        return self._negative_entry.get_text()

    def get_raw_positive_prompt(self) -> str:
        """Get just the user-entered positive prompt (without manager words)."""
        return self._positive_entry.get_text()

    def get_refiner_prompt(self) -> str:
        """Get the refiner prompt text, prepending checked words from manager."""
        user_prompt = self._refiner_entry.get_text()
        checked_words = self._prompt_manager.get_checked_words_string()

        if checked_words and user_prompt:
            return f"{checked_words}, {user_prompt}"
        elif checked_words:
            return checked_words
        else:
            return user_prompt

    def set_refiner_prompt(self, text: str):
        """Set the refiner prompt text."""
        self._refiner_entry.set_text(text)

    def set_refiner_mode(self, enabled: bool):
        """Show or hide the refiner prompt entry."""
        self._refiner_entry.set_visible(enabled)

    def clear(self):
        """Clear both prompts."""
        self._positive_entry.clear()
        self._negative_entry.clear()

    def set_positive_prompt(self, text: str):
        """Set the positive prompt text."""
        self._positive_entry.set_text(text)

    def set_negative_prompt(self, text: str):
        """Set the negative prompt text."""
        self._negative_entry.set_text(text)

    def set_prompts(self, positive: str, negative: str):
        """Set both prompts at once."""
        self._positive_entry.set_text(positive)
        self._negative_entry.set_text(negative)

    def get_paned_positions(self) -> dict:
        """Get the current paned positions for saving."""
        return {
            "prompts_width": self._main_paned.get_position(),
            "prompts_split": self._prompts_paned.get_position(),
        }

    def set_paned_positions(self, positions: dict):
        """Restore paned positions."""
        if "prompts_width" in positions:
            self._main_paned.set_position(positions["prompts_width"])
        if "prompts_split" in positions:
            self._prompts_paned.set_position(positions["prompts_split"])
            self._prompts_split_restored = True

    def get_font_sizes(self) -> dict:
        """Get the current font sizes for saving."""
        return {
            "positive": self._positive_entry.get_font_size(),
            "negative": self._negative_entry.get_font_size(),
        }

    def set_font_sizes(self, positive_size: int, negative_size: int):
        """Restore font sizes."""
        if positive_size > 0:
            self._positive_entry.set_font_size(positive_size)
        if negative_size > 0:
            self._negative_entry.set_font_size(negative_size)
