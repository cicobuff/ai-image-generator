"""Configuration management for AI Image Generator."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml

from src.utils.constants import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    DEFAULT_STEPS,
    DEFAULT_CFG_SCALE,
    DEFAULT_SEED,
)


@dataclass
class DirectoriesConfig:
    """Configuration for directory paths."""
    models: str = "./models"
    output: str = "./output"


@dataclass
class GPUConfig:
    """Configuration for GPU selection."""
    selected: list[int] = field(default_factory=lambda: [0])


@dataclass
class GenerationConfig:
    """Configuration for default generation parameters."""
    default_width: int = DEFAULT_WIDTH
    default_height: int = DEFAULT_HEIGHT
    default_sampler: str = DEFAULT_SAMPLER
    default_scheduler: str = DEFAULT_SCHEDULER
    default_steps: int = DEFAULT_STEPS
    default_cfg_scale: float = DEFAULT_CFG_SCALE
    default_seed: int = DEFAULT_SEED


@dataclass
class WindowConfig:
    """Configuration for window and panel sizes."""
    width: int = 1400
    height: int = 900
    maximized: bool = False
    # Panel positions (Gtk.Paned positions)
    left_panel_width: int = 280
    right_panel_position: int = 800
    center_panel_height: int = 500
    # Prompt section paned positions
    prompt_section_width: int = -1  # -1 means auto (use default)
    prompt_section_split: int = -1  # -1 means auto (equal split)
    prompt_manager_split: int = -1  # -1 means auto (equal split)
    # Prompt font sizes (in points, 0 means use default)
    positive_prompt_font_size: int = 0
    negative_prompt_font_size: int = 0
    refiner_prompt_font_size: int = 0
    # Panel collapsed states
    left_panel_collapsed: bool = False
    right_panel_collapsed: bool = False
    prompt_panel_collapsed: bool = False


@dataclass
class OutpaintConfig:
    """Configuration for outpainting settings."""
    # Edge zone size in pixels - area near image edge where outpaint mask can be drawn
    edge_zone_size: int = 200
    # Default extension size for outpaint masks
    default_extension: int = 256


@dataclass
class AppConfig:
    """Main application configuration."""
    version: str = "1.0"
    directories: DirectoriesConfig = field(default_factory=DirectoriesConfig)
    gpus: GPUConfig = field(default_factory=GPUConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    outpaint: OutpaintConfig = field(default_factory=OutpaintConfig)

    def to_dict(self) -> dict:
        """Convert config to dictionary for YAML serialization."""
        return {
            "version": self.version,
            "directories": asdict(self.directories),
            "gpus": asdict(self.gpus),
            "generation": asdict(self.generation),
            "window": asdict(self.window),
            "outpaint": asdict(self.outpaint),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create config from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            directories=DirectoriesConfig(**data.get("directories", {})),
            gpus=GPUConfig(**data.get("gpus", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            window=WindowConfig(**data.get("window", {})),
            outpaint=OutpaintConfig(**data.get("outpaint", {})),
        )

    def get_models_path(self) -> Path:
        """Get absolute path to models directory."""
        path = Path(self.directories.models)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def get_output_path(self) -> Path:
        """Get absolute path to output directory."""
        path = Path(self.directories.output)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()


class ConfigManager:
    """Manages loading and saving application configuration."""

    def __init__(self):
        self._config: Optional[AppConfig] = None

    @property
    def config(self) -> AppConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def exists(self) -> bool:
        """Check if configuration file exists."""
        return CONFIG_FILE.exists()

    def load(self) -> AppConfig:
        """Load configuration from file, or return defaults."""
        if not CONFIG_FILE.exists():
            return AppConfig()

        try:
            with open(CONFIG_FILE, "r") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return AppConfig()
                return AppConfig.from_dict(data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return AppConfig()

    def save(self, config: Optional[AppConfig] = None) -> bool:
        """Save configuration to file."""
        if config is not None:
            self._config = config

        if self._config is None:
            self._config = AppConfig()

        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                yaml.dump(
                    self._config.to_dict(),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def update(self, **kwargs) -> None:
        """Update specific configuration values."""
        config = self.config

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.directories, key):
                setattr(config.directories, key, value)
            elif hasattr(config.gpus, key):
                setattr(config.gpus, key, value)
            elif hasattr(config.generation, key):
                setattr(config.generation, key, value)

        self.save()

    def ensure_directories(self) -> None:
        """Ensure model and output directories exist."""
        self.config.get_models_path().mkdir(parents=True, exist_ok=True)
        self.config.get_output_path().mkdir(parents=True, exist_ok=True)


# Global config manager instance
config_manager = ConfigManager()
