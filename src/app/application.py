"""Main GTK Application class."""

import sys
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gio, GLib, Gdk

from src.utils.constants import APP_ID, APP_NAME
from src.core.config import config_manager
from src.core.gpu_manager import gpu_manager


class AIImageGeneratorApp(Gtk.Application):
    """Main application class for AI Image Generator."""

    def __init__(self):
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )
        self._window = None
        self._setup_screen = None
        self._work_screen = None
        self._app_icon_texture = None

    def do_startup(self):
        """Called when the application starts up."""
        Gtk.Application.do_startup(self)

        # Load CSS
        self._load_css()

        # Load app icon
        self._load_app_icon()

        # Initialize GPU manager
        gpu_manager.initialize()

        # Create actions
        self._create_actions()

    def do_activate(self):
        """Called when the application is activated."""
        if not self._window:
            self._window = Gtk.ApplicationWindow(application=self)
            self._window.set_title(APP_NAME)

            # Set the application icon
            self._window.set_icon_name(APP_ID)

            # Restore window size from config
            window_config = config_manager.config.window
            self._window.set_default_size(window_config.width, window_config.height)
            if window_config.maximized:
                self._window.maximize()

            # Handle window close to ensure clean exit with torch.compile
            self._window.connect("close-request", self._on_close_request)

            # Show appropriate screen based on config existence
            if config_manager.exists():
                self._show_work_screen()
            else:
                self._show_setup_screen()

        self._window.present()

    def _on_close_request(self, window):
        """Handle window close request - ensures clean exit with torch.compile."""
        import os

        # Save window size and panel positions
        try:
            self._save_window_state()
        except Exception as e:
            print(f"Error saving window state: {e}")

        # Quick cleanup
        try:
            from src.backends.diffusers_backend import diffusers_backend
            diffusers_backend.unload_model()
        except Exception:
            pass

        try:
            from src.backends.upscale_backend import upscale_backend
            upscale_backend.unload_model()
        except Exception:
            pass

        try:
            gpu_manager.shutdown()
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Force immediate exit to avoid torch.compile hang
        os._exit(0)
        return True  # Won't be reached, but indicates we handled the event

    def _save_window_state(self):
        """Save window size and panel positions to config."""
        window_config = config_manager.config.window

        # Check if maximized
        window_config.maximized = self._window.is_maximized()

        # Only save size if not maximized
        if not window_config.maximized:
            window_config.width = self._window.get_width()
            window_config.height = self._window.get_height()

        # Get panel positions from work screen if available
        if self._work_screen is not None:
            panel_positions = self._work_screen.get_panel_positions()
            window_config.left_panel_width = panel_positions.get("left", 280)
            window_config.right_panel_position = panel_positions.get("right", 800)
            window_config.center_panel_height = panel_positions.get("center", 500)
            # Prompt section positions
            window_config.prompt_section_width = panel_positions.get("prompt_section_width", -1)
            window_config.prompt_section_split = panel_positions.get("prompt_section_split", -1)
            window_config.prompt_manager_split = panel_positions.get("prompt_manager_split", -1)
            # Prompt font sizes
            window_config.positive_prompt_font_size = panel_positions.get("positive_prompt_font_size", 0)
            window_config.negative_prompt_font_size = panel_positions.get("negative_prompt_font_size", 0)
            # Panel collapsed states
            window_config.left_panel_collapsed = panel_positions.get("left_panel_collapsed", False)
            window_config.right_panel_collapsed = panel_positions.get("right_panel_collapsed", False)
            window_config.prompt_panel_collapsed = panel_positions.get("prompt_panel_collapsed", False)

        config_manager.save()

    def do_shutdown(self):
        """Called when the application shuts down."""
        import os

        # Cleanup diffusers backend (unload models, clear CUDA cache)
        try:
            from src.backends.diffusers_backend import diffusers_backend
            diffusers_backend.unload_model()
        except Exception as e:
            print(f"Error unloading diffusers model: {e}")

        # Cleanup upscale backend
        try:
            from src.backends.upscale_backend import upscale_backend
            upscale_backend.unload_model()
        except Exception as e:
            print(f"Error unloading upscale model: {e}")

        # Cleanup GPU manager
        gpu_manager.shutdown()

        # Basic CUDA cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Call parent shutdown
        try:
            Gtk.Application.do_shutdown(self)
        except Exception:
            pass

        # Force exit to avoid hanging on torch.compile cleanup
        # This is necessary because CUDA graphs from torch.compile
        # can prevent clean process termination
        os._exit(0)

    def _load_css(self):
        """Load application CSS styles."""
        css_path = Path(__file__).parent.parent / "ui" / "resources" / "style.css"

        if css_path.exists():
            css_provider = Gtk.CssProvider()
            css_provider.load_from_path(str(css_path))

            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(),
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )

    def _load_app_icon(self):
        """Load and set the application icon."""
        icons_path = Path(__file__).parent.parent / "ui" / "resources" / "icons"

        if icons_path.exists():
            try:
                # Add our icons directory to the icon theme search path
                icon_theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())
                icon_theme.add_search_path(str(icons_path))

                # Also load the texture for direct use if needed
                icon_file_path = icons_path / "hicolor" / "scalable" / "apps" / f"{APP_ID}.svg"
                if icon_file_path.exists():
                    icon_file = Gio.File.new_for_path(str(icon_file_path))
                    self._app_icon_texture = Gdk.Texture.new_from_file(icon_file)
            except Exception as e:
                print(f"Error loading app icon: {e}")

    def _create_actions(self):
        """Create application actions."""
        # Quit action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self._on_quit)
        self.add_action(quit_action)

        # About action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self._on_about)
        self.add_action(about_action)

        # Settings action
        settings_action = Gio.SimpleAction.new("settings", None)
        settings_action.connect("activate", self._on_settings)
        self.add_action(settings_action)

        # Set keyboard shortcuts
        self.set_accels_for_action("app.quit", ["<Control>q"])

    def _on_quit(self, action, param):
        """Handle quit action."""
        self.quit()

    def _on_about(self, action, param):
        """Show about dialog."""
        about = Gtk.AboutDialog(
            transient_for=self._window,
            modal=True,
            program_name=APP_NAME,
            version="0.1.0",
            authors=["AI Image Generator Team"],
            copyright="2024",
            license_type=Gtk.License.MIT_X11,
            comments="A GTK 4 AI Image Generator using Stable Diffusion",
        )
        about.present()

    def _on_settings(self, action, param):
        """Show settings / setup screen."""
        self._show_setup_screen()

    def _show_setup_screen(self):
        """Show the setup/configuration screen."""
        # Import here to avoid circular imports
        from src.ui.screens.setup_screen import SetupScreen

        if self._setup_screen is None:
            self._setup_screen = SetupScreen(self._on_setup_complete)

        self._window.set_child(self._setup_screen)

    def _show_work_screen(self):
        """Show the main work screen."""
        # Import here to avoid circular imports
        from src.ui.screens.work_screen import WorkScreen

        if self._work_screen is None:
            self._work_screen = WorkScreen()

        self._window.set_child(self._work_screen)

    def _on_setup_complete(self):
        """Called when setup is complete."""
        # Ensure directories exist
        config_manager.ensure_directories()

        # Switch to work screen
        self._show_work_screen()

    def get_window(self) -> Gtk.ApplicationWindow:
        """Get the main application window."""
        return self._window


def run_app():
    """Run the application."""
    app = AIImageGeneratorApp()
    return app.run(sys.argv)
