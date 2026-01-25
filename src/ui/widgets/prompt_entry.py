"""Prompt entry widgets with colored borders."""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Pango

# Font size constants
DEFAULT_FONT_SIZE = 10  # Default font size in points
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 24
FONT_SIZE_STEP = 1


class PromptEntry(Gtk.Box):
    """Text entry widget for prompts with colored border."""

    def __init__(self, label: str, is_positive: bool = True, placeholder: str = "", style_type: str = None):
        """
        Create a prompt entry widget.

        Args:
            label: The label text for the entry
            is_positive: If True, use green style; if False, use red style (ignored if style_type is set)
            placeholder: Placeholder text (not currently used)
            style_type: Explicit style type: "positive", "negative", or "refiner"
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        # Determine style type
        if style_type:
            self._style_type = style_type
        else:
            self._style_type = "positive" if is_positive else "negative"
        self._font_size = DEFAULT_FONT_SIZE

        self._build_ui(label, placeholder)

    def _build_ui(self, label_text: str, placeholder: str):
        """Build the widget UI."""
        # Header row with label and font size controls
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.append(header)

        # Label
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_hexpand(True)
        label.add_css_class(f"prompt-label-{self._style_type}")
        header.append(label)

        # Font size decrease button
        self._decrease_btn = Gtk.Button.new_from_icon_name("zoom-out-symbolic")
        self._decrease_btn.add_css_class("flat")
        self._decrease_btn.add_css_class("prompt-font-btn")
        self._decrease_btn.set_tooltip_text("Decrease font size")
        self._decrease_btn.connect("clicked", self._on_decrease_font)
        header.append(self._decrease_btn)

        # Font size increase button
        self._increase_btn = Gtk.Button.new_from_icon_name("zoom-in-symbolic")
        self._increase_btn.add_css_class("flat")
        self._increase_btn.add_css_class("prompt-font-btn")
        self._increase_btn.set_tooltip_text("Increase font size")
        self._increase_btn.connect("clicked", self._on_increase_font)
        header.append(self._increase_btn)

        # Frame for border styling
        frame = Gtk.Frame()
        frame.add_css_class(f"prompt-{self._style_type}")
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

    def _on_increase_font(self, button):
        """Increase font size."""
        if self._font_size < MAX_FONT_SIZE:
            self.set_font_size(self._font_size + FONT_SIZE_STEP)

    def _on_decrease_font(self, button):
        """Decrease font size."""
        if self._font_size > MIN_FONT_SIZE:
            self.set_font_size(self._font_size - FONT_SIZE_STEP)

    def _update_button_sensitivity(self):
        """Update button sensitivity based on current font size."""
        self._decrease_btn.set_sensitive(self._font_size > MIN_FONT_SIZE)
        self._increase_btn.set_sensitive(self._font_size < MAX_FONT_SIZE)

    def _apply_font_size(self):
        """Apply the current font size to the text view."""
        font_desc = Pango.FontDescription()
        font_desc.set_size(self._font_size * Pango.SCALE)

        # Get or create a CSS provider for this text view
        css = f"textview {{ font-size: {self._font_size}pt; }}"
        provider = Gtk.CssProvider()
        provider.load_from_data(css.encode())

        # Apply to the text view
        self._text_view.get_style_context().add_provider(
            provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self._update_button_sensitivity()

    def get_font_size(self) -> int:
        """Get the current font size."""
        return self._font_size

    def set_font_size(self, size: int):
        """Set the font size."""
        self._font_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, size))
        self._apply_font_size()

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
