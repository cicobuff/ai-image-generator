"""Prompt entry widgets with colored borders."""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk


class PromptEntry(Gtk.Box):
    """Text entry widget for prompts with colored border."""

    def __init__(self, label: str, is_positive: bool = True, placeholder: str = ""):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._is_positive = is_positive

        self._build_ui(label, placeholder)

    def _build_ui(self, label_text: str, placeholder: str):
        """Build the widget UI."""
        # Label
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        if self._is_positive:
            label.add_css_class("prompt-label-positive")
        else:
            label.add_css_class("prompt-label-negative")
        self.append(label)

        # Frame for border styling
        frame = Gtk.Frame()
        if self._is_positive:
            frame.add_css_class("prompt-positive")
        else:
            frame.add_css_class("prompt-negative")
        self.append(frame)

        # Scrolled window for text view
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(80)
        scrolled.set_vexpand(True)
        frame.set_child(scrolled)

        # Text view
        self._text_view = Gtk.TextView()
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._text_view.set_accepts_tab(False)

        if placeholder:
            # GTK 4 doesn't have native placeholder for TextView
            # We'll handle this with buffer management if needed
            pass

        scrolled.set_child(self._text_view)

    def get_text(self) -> str:
        """Get the current text."""
        buffer = self._text_view.get_buffer()
        start = buffer.get_start_iter()
        end = buffer.get_end_iter()
        return buffer.get_text(start, end, False)

    def set_text(self, text: str):
        """Set the text content."""
        buffer = self._text_view.get_buffer()
        buffer.set_text(text)

    def clear(self):
        """Clear the text content."""
        self.set_text("")

    def get_buffer(self) -> Gtk.TextBuffer:
        """Get the underlying text buffer."""
        return self._text_view.get_buffer()


class PromptPanel(Gtk.Box):
    """Panel containing both positive and negative prompt entries."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        self._build_ui()

    def _build_ui(self):
        """Build the prompt panel UI."""
        # Positive prompt
        self._positive_entry = PromptEntry(
            label="Positive Prompt",
            is_positive=True,
            placeholder="Describe what you want to generate...",
        )
        self._positive_entry.set_vexpand(True)
        self.append(self._positive_entry)

        # Negative prompt
        self._negative_entry = PromptEntry(
            label="Negative Prompt",
            is_positive=False,
            placeholder="Describe what you want to avoid...",
        )
        self._negative_entry.set_vexpand(True)
        self.append(self._negative_entry)

    @property
    def positive_entry(self) -> PromptEntry:
        """Get the positive prompt entry."""
        return self._positive_entry

    @property
    def negative_entry(self) -> PromptEntry:
        """Get the negative prompt entry."""
        return self._negative_entry

    def get_positive_prompt(self) -> str:
        """Get the positive prompt text."""
        return self._positive_entry.get_text()

    def get_negative_prompt(self) -> str:
        """Get the negative prompt text."""
        return self._negative_entry.get_text()

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
