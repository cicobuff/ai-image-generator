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

    def do_startup(self):
        """Called when the application starts up."""
        Gtk.Application.do_startup(self)

        # Load CSS
        self._load_css()

        # Initialize GPU manager
        gpu_manager.initialize()

        # Create actions
        self._create_actions()

    def do_activate(self):
        """Called when the application is activated."""
        if not self._window:
            self._window = Gtk.ApplicationWindow(application=self)
            self._window.set_title(APP_NAME)
            self._window.set_default_size(1400, 900)

            # Show appropriate screen based on config existence
            if config_manager.exists():
                self._show_work_screen()
            else:
                self._show_setup_screen()

        self._window.present()

    def do_shutdown(self):
        """Called when the application shuts down."""
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

        # Final CUDA cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        Gtk.Application.do_shutdown(self)

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
