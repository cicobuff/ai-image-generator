"""Collapsible panel functionality for Gtk.Paned layouts."""

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk


class PanedCollapseButton(Gtk.Button):
    """A button that collapses/expands a paned child."""

    def __init__(
        self,
        paned: Gtk.Paned,
        collapse_direction: str,  # "left", "right", or "bottom"
        initially_collapsed: bool = False,
        collapsed_size: int = 28,  # Size when collapsed (just the button)
    ):
        """
        Initialize the collapse button.

        Args:
            paned: The Gtk.Paned to control
            collapse_direction: Which direction to collapse
            initially_collapsed: Whether to start collapsed
            collapsed_size: Size in pixels when collapsed
        """
        super().__init__()

        self._paned = paned
        self._collapse_direction = collapse_direction
        self._collapsed = False
        self._collapsed_size = collapsed_size
        self._saved_position = None

        # Set up button appearance
        self.add_css_class("flat")

        if collapse_direction == "bottom":
            self.add_css_class("collapse-toggle-bottom")
            self._expand_icon = "go-down-symbolic"
            self._collapse_icon = "go-up-symbolic"
        elif collapse_direction == "left":
            self.add_css_class("collapse-toggle")
            self._expand_icon = "go-next-symbolic"
            self._collapse_icon = "go-previous-symbolic"
        else:  # right
            self.add_css_class("collapse-toggle")
            self._expand_icon = "go-previous-symbolic"
            self._collapse_icon = "go-next-symbolic"

        self.connect("clicked", self._on_clicked)

        # Set initial state after paned is realized
        if initially_collapsed:
            self._paned.connect("realize", lambda w: self._set_collapsed(True))

        self._update_icon()

    def _update_icon(self):
        """Update button icon based on state."""
        if self._collapsed:
            self.set_icon_name(self._expand_icon)
            self.set_tooltip_text("Expand panel")
        else:
            self.set_icon_name(self._collapse_icon)
            self.set_tooltip_text("Collapse panel")

    def _on_clicked(self, button):
        """Handle button click."""
        self._set_collapsed(not self._collapsed)

    def _set_collapsed(self, collapsed: bool):
        """Set the collapsed state."""
        if collapsed == self._collapsed:
            return

        if collapsed:
            # Save current position before collapsing
            self._saved_position = self._paned.get_position()

            # Calculate collapsed position based on direction
            if self._collapse_direction == "left":
                # Collapse to left edge
                self._paned.set_position(self._collapsed_size)
            elif self._collapse_direction == "right":
                # Collapse to right edge - get total width and set position near end
                width = self._paned.get_allocated_width()
                self._paned.set_position(width - self._collapsed_size)
            else:  # bottom
                # Collapse to bottom - get total height and set position near end
                height = self._paned.get_allocated_height()
                self._paned.set_position(height - self._collapsed_size)
        else:
            # Restore saved position
            if self._saved_position is not None:
                self._paned.set_position(self._saved_position)

        self._collapsed = collapsed
        self._update_icon()

    @property
    def collapsed(self) -> bool:
        """Get collapsed state."""
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value: bool):
        """Set collapsed state."""
        self._set_collapsed(value)

    def set_saved_position(self, position: int):
        """Set the saved position (used when restoring from config)."""
        self._saved_position = position
