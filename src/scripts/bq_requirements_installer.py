#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 🐾 Beastique Requirements Installer
Date: May 8, 2025
Version: 1.0.0
Author: Dennis 'dendogg' Smaltz

✒ Description;
    A hyper-optimized, feature-rich Python package installer with advanced capabilities.
    Beastique makes it easy to install multiple packages in the optimal order with 
    dependency resolution, parallel installation, and extensive verification.


✒ Key Features:
    - Concurrent installation with configurable worker count
    - Smart dependency resolution and topological sorting
    - Category-based package organization and filtering
    - Progress bars and detailed logging
    - Package import verification
    - Resumable installations with state tracking
    - Customizable installation settings
    - Detailed summary reports


✒ Usage Instructions:
    python bq_req.py [options]
    
    Options:
    -c, --categories      Specify categories to install (comma-separated)
    -p, --packages        Specify individual packages to install (comma-separated)
    --force-reinstall     Force reinstallation of packages
    --workers N           Set number of parallel workers (default: CPU-based)
    --timeout N           Set installation timeout in seconds
    --list-categories     Show available package categories
    --list-packages       Show all available packages with details
    --check-only          Show what would be installed without installing
    --include-optional    Include optional packages in installation


✒ ️Examples:
    # Install all default packages
    python bq_req.py
    
    # Install only Image & Color Analysis packages
    python bq_req.py -c "Image & Color Analysis"
    
    # Install specific packages
    python bq_req.py -p numpy,pandas,matplotlib
    
    # List all available package categories
    python bq_req.py --list-categories


✒ Other Important Information...
    - The installer creates a detailed log file at beastique_install.log
    - Installation state is saved to beastique_install_state.json
    - System dependencies may be required for some packages
    - Installation order is optimized to minimize conflicts
    - The installer supports virtual environments

---------
"""

import subprocess
import sys
import time
import platform
import concurrent.futures
import os
import re
import signal
import shutil
import json
import hashlib
import argparse
import urllib.request
from typing import List, Dict, Tuple, Optional, Set, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# ====== CONFIGURATION ======
class Config:
    """Global configuration settings."""
    # File paths
    LOG_FILE = Path("beastique_install.log")
    REQUIREMENTS_OUT = Path("beastique_requirements.txt")
    INSTALL_STATE_FILE = Path("beastique_install_state.json")
    
    # Performance settings
    MAX_WORKERS = min(os.cpu_count() or 4, 8)  # Sensible default based on CPUs
    TIMEOUT = 300  # Seconds
    RETRY_COUNT = 2
    RETRY_DELAY = 5  # Seconds
    
    # Network settings
    PIP_INDEX_URL = None  # Set to custom PyPI mirror if needed
    PIP_EXTRA_INDEX_URL = None
    NETWORK_TIMEOUT = 30  # Seconds
    
    # Feature flags
    CHECK_HASH = True
    VERIFY_IMPORTS = True
    INSTALL_DEPS_FIRST = True
    FORCE_REINSTALL = False
    USE_WHEELHOUSE = False
    WHEELHOUSE_PATH = Path("wheelhouse")
    PROGRESS_BAR = True
    
    # UI settings
    SHOW_CATEGORY_STATS = True
    DETAILED_SUMMARY = True
    SILENT_MODE = False
    DEBUG_MODE = False


# ====== PACKAGE DEFINITIONS ======
@dataclass
class PackageInfo:
    """Detailed package information."""
    name: str
    priority: int = 0  # Higher means install earlier
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    alternative_names: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    extras: List[str] = field(default_factory=list)
    system_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    optional: bool = False
    purpose: str = ""
    url: str = ""
    hash: str = ""  # Expected hash for verification
    
    def get_install_spec(self) -> str:
        """Get the pip install specification for this package."""
        spec = self.name
        if self.extras:
            spec += f"[{','.join(self.extras)}]"
        if self.min_version and self.max_version:
            spec += f">={self.min_version},<={self.max_version}"
        elif self.min_version:
            spec += f">={self.min_version}"
        elif self.max_version:
            spec += f"<={self.max_version}"
        return spec


class PackageStatus(Enum):
    """Possible states for a package installation."""
    PENDING = "pending"
    INSTALLING = "installing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


@dataclass
class InstallResult:
    """Result of an installation attempt."""
    package: str
    status: PackageStatus
    version: Optional[str] = None
    message: str = ""
    duration: float = 0.0
    attempt: int = 1
    stdout: str = ""
    stderr: str = ""
    timestamp: float = field(default_factory=time.time)


# ====== TERMINAL UTILITIES ======
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    GRAY = "\033[90m"
    
    @staticmethod
    def supports_color() -> bool:
        """Check if the terminal supports color."""
        plat = sys.platform
        supported_platform = (plat != 'win32' or 'ANSICON' in os.environ)
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        return supported_platform and is_a_tty


class ProgressBar:
    """Advanced progress bar with ETA and speed estimation."""
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.start_time = time.time()
        self.completed = 0
        self.speeds = []
        
    def update(self, completed: int) -> None:
        """Update the progress bar."""
        if not Config.PROGRESS_BAR or Config.SILENT_MODE:
            return
            
        self.completed = completed
        
        # Calculate progress
        progress = min(completed / self.total, 1.0)
        filled_width = int(self.width * progress)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        rate = completed / elapsed if elapsed > 0 else 0
        
        # Keep track of speed for smoothing
        self.speeds.append(rate)
        if len(self.speeds) > 10:
            self.speeds.pop(0)
        avg_speed = sum(self.speeds) / len(self.speeds)
        
        # Calculate ETA
        remaining = (self.total - completed) / avg_speed if avg_speed > 0 else 0
        
        # Format strings
        bar = '█' * filled_width + '░' * (self.width - filled_width)
        percent = int(progress * 100)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
        
        # Print progress bar
        sys.stdout.write(f"\r{Colors.BLUE}Progress: {Colors.ENDC}[{bar}] {percent}% | "
                        f"{completed}/{self.total} | "
                        f"Rate: {avg_speed:.2f} pkg/s | "
                        f"ETA: {eta_str}{' ' * 10}")
        sys.stdout.flush()
        
    def finish(self) -> None:
        """Complete the progress bar."""
        if not Config.PROGRESS_BAR or Config.SILENT_MODE:
            return
            
        sys.stdout.write("\n")
        sys.stdout.flush()


class Logger:
    """Advanced logger with levels and formatting."""
    LEVELS = {
        "DEBUG": (Colors.GRAY, 0),
        "INFO": (Colors.BLUE, 1),
        "SUCCESS": (Colors.GREEN, 1),
        "WARNING": (Colors.YELLOW, 2),
        "ERROR": (Colors.RED, 2),
        "CRITICAL": (f"{Colors.RED}{Colors.BOLD}", 3)
    }
    
    current_level = 1  # Default level (INFO)
    log_file = Config.LOG_FILE
    
    @classmethod
    def log(cls, message: str, level: str = "INFO", console: bool = True) -> None:
        """Log a message to file and possibly console."""
        if level not in cls.LEVELS:
            level = "INFO"
        
        color, importance = cls.LEVELS[level]
        
        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Create log entry
        log_entry = f"[{timestamp}] {level}: {message}"
        colored_entry = f"{color}[{timestamp}] {level}:{Colors.ENDC} {message}"
        
        # Write to log file
        with cls.log_file.open("a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        
        # Print to console if appropriate
        if console and importance >= cls.current_level and not Config.SILENT_MODE:
            print(colored_entry)
    
    @classmethod
    def debug(cls, message: str) -> None:
        """Log a debug message."""
        if Config.DEBUG_MODE:
            cls.log(message, "DEBUG")
    
    @classmethod
    def info(cls, message: str) -> None:
        """Log an info message."""
        cls.log(message, "INFO")
    
    @classmethod
    def success(cls, message: str) -> None:
        """Log a success message."""
        cls.log(message, "SUCCESS")
    
    @classmethod
    def warning(cls, message: str) -> None:
        """Log a warning message."""
        cls.log(message, "WARNING")
    
    @classmethod
    def error(cls, message: str) -> None:
        """Log an error message."""
        cls.log(message, "ERROR")
    
    @classmethod
    def critical(cls, message: str) -> None:
        """Log a critical message."""
        cls.log(message, "CRITICAL")


# ====== PACKAGE DATA ======
# Core packages with dependencies and priorities
PACKAGES = {
    "Core": [
        PackageInfo(
            name="numpy",
            priority=100,
            min_version="1.22.0",
            purpose="Numerical computing foundation",
            dependencies=[]
        ),
        PackageInfo(
            name="pandas",
            priority=90,
            min_version="1.4.0",
            purpose="Data manipulation and analysis",
            dependencies=["numpy"]
        ),
        PackageInfo(
            name="scipy",
            priority=90,
            min_version="1.8.0",
            purpose="Scientific computing",
            dependencies=["numpy"]
        ),
        PackageInfo(
            name="matplotlib",
            priority=80,
            min_version="3.5.0",
            purpose="Data visualization",
            dependencies=["numpy"]
        ),
        PackageInfo(
            name="seaborn",
            min_version="0.11.0",
            purpose="Statistical data visualization",
            dependencies=["matplotlib", "pandas", "numpy"]
        ),
        PackageInfo(
            name="opencv-python-headless",
            min_version="4.6.0",
            purpose="Computer vision library (headless)",
            dependencies=["numpy"]
        ),
        PackageInfo(
            name="pillow",
            priority=70,
            min_version="9.0.0",
            purpose="Image processing",
            alternative_names=["PIL"]
        ),
        PackageInfo(
            name="scikit-learn",
            min_version="1.0.0",
            purpose="Machine learning",
            dependencies=["numpy", "scipy"]
        ),
        PackageInfo(name="tqdm", min_version="4.64.0", purpose="Progress bars"),
        PackageInfo(name="rich", min_version="12.0.0", purpose="Rich terminal output"),
        PackageInfo(name="colorama", purpose="Cross-platform colored terminal output"),
        PackageInfo(name="termcolor", purpose="Terminal color formatting"),
        PackageInfo(name="loguru", min_version="0.6.0", purpose="Logging"),
        PackageInfo(name="python-dotenv", purpose="Environment variable management"),
        PackageInfo(name="pyyaml", min_version="6.0", purpose="YAML parsing"),
        PackageInfo(name="json5", purpose="JSON5 parsing"),
        PackageInfo(name="regex", purpose="Enhanced regular expressions"),
        PackageInfo(name="filetype", purpose="File type identification"),
        PackageInfo(name="pyperclip", purpose="Clipboard access"),
        PackageInfo(name="pyfiglet", purpose="ASCII art text"),
        PackageInfo(name="ascii-magic", purpose="ASCII art conversion"),
        PackageInfo(name="ascii-math", purpose="ASCII math notation"),
        PackageInfo(name="typer", purpose="CLI building", dependencies=["click"]),
        PackageInfo(
            name="click",
            min_version="8.0.0",
            purpose="Command line interfaces"
        ),
        PackageInfo(name="prompt-toolkit", purpose="Interactive command line interfaces"),
        PackageInfo(name="tabulate", purpose="Pretty-print tabular data"),
        PackageInfo(name="sh", purpose="Subprocess interface")
    ],
    
    "Image & Color Analysis": [
        PackageInfo(name="webcolors", purpose="Color name conversion"),
        PackageInfo(name="colour-science", purpose="Color science algorithms", dependencies=["numpy"]),
        PackageInfo(name="colormath", purpose="Color math and conversion"),
        PackageInfo(name="imutils", purpose="Image processing utilities", dependencies=["opencv-python-headless"]),
        PackageInfo(name="scikit-image", purpose="Image processing", dependencies=["numpy", "scipy"]),
        PackageInfo(name="imagehash", purpose="Perceptual image hashing", dependencies=["pillow", "numpy"]),
        PackageInfo(name="dominant-color-extractor", purpose="Extract dominant colors"),
        PackageInfo(name="cssutils", purpose="CSS parser and builder"),
        PackageInfo(
            name="opencv-contrib-python-headless",
            purpose="OpenCV with contrib modules (headless)",
            dependencies=["numpy"]
        ),
        PackageInfo(name="removebg", purpose="Remove image backgrounds"),
        PackageInfo(name="pillow-heif", purpose="HEIF image support", dependencies=["pillow"])
    ],
    
    "Document Handling": [
        PackageInfo(name="markdown2", purpose="Markdown parser"),
        PackageInfo(name="html2text", purpose="HTML to text conversion"),
        PackageInfo(name="docx2txt", purpose="DOCX to text conversion"),
        PackageInfo(name="openpyxl", purpose="Excel file handling"),
        PackageInfo(name="xlrd", purpose="Excel file reading"),
        PackageInfo(name="pyxlsb", purpose="Excel Binary Workbook files"),
        PackageInfo(name="csvkit", purpose="CSV utilities"),
        PackageInfo(name="xlutils", purpose="Excel utilities", dependencies=["xlrd"])
    ],
    
    "Steganography & Encryption": [
        PackageInfo(name="cryptography", min_version="37.0.0", purpose="Cryptography"),
        PackageInfo(name="pycryptodome", purpose="Cryptographic algorithms"),
        PackageInfo(name="pyAesCrypt", purpose="AES file encryption"),
        PackageInfo(name="steganocryptopy", purpose="Steganography with cryptography"),
        PackageInfo(name="stegano", purpose="Steganography"),
        PackageInfo(name="zxcvbn", purpose="Password strength estimation"),
        PackageInfo(name="password-strength", purpose="Password strength evaluation"),
        PackageInfo(name="hashids", purpose="Generate short unique ids from integers")
    ],
    
    "NLP, Language & Scraping": [
        PackageInfo(name="nltk", purpose="Natural language toolkit"),
        PackageInfo(name="textblob", purpose="Text processing", dependencies=["nltk"]),
        PackageInfo(name="langdetect", purpose="Language detection"),
        PackageInfo(name="newspaper3k", purpose="Article scraping & curation"),
        PackageInfo(name="readability-lxml", purpose="Web page readability"),
        PackageInfo(name="beautifulsoup4", purpose="HTML/XML parsing"),
        PackageInfo(name="lxml", purpose="XML and HTML processing"),
        PackageInfo(name="httpx", purpose="HTTP client"),
        PackageInfo(name="requests", purpose="HTTP requests"),
        PackageInfo(name="fake-useragent", purpose="User agent rotation")
    ],
    
    "Geo Tools": [
        PackageInfo(name="shapely", purpose="Geometric objects"),
        PackageInfo(name="pyshp", purpose="ESRI Shapefile reading"),
        PackageInfo(
            name="geopandas",
            purpose="Geospatial data operations",
            dependencies=["pandas", "shapely", "fiona"]
        ),
        PackageInfo(name="folium", purpose="Interactive maps", dependencies=["branca"]),
        PackageInfo(name="fiona", purpose="Geospatial data reading/writing"),
        PackageInfo(name="rasterio", purpose="Geospatial raster data"),
        PackageInfo(
            name="geoplot",
            purpose="Geospatial data visualization",
            dependencies=["geopandas", "matplotlib"]
        ),
        PackageInfo(name="cartopy", purpose="Cartographic tools", dependencies=["numpy", "matplotlib"]),
        PackageInfo(name="geojson", purpose="GeoJSON encoding/decoding"),
        PackageInfo(name="mercantile", purpose="Web mercator tile calculations"),
        PackageInfo(name="mapclassify", purpose="Classification schemes for choropleth maps"),
        PackageInfo(name="elevation", purpose="Digital elevation models"),
        PackageInfo(name="contextily", purpose="Web map tile backgrounds"),
        PackageInfo(name="branca", purpose="ColorBrewer colors for maps"),
        PackageInfo(name="pyproj", purpose="Cartographic projections")
    ],
    
    "SVG & Rendering": [
        PackageInfo(name="svgwrite", purpose="SVG creation"),
        PackageInfo(name="cairosvg", purpose="SVG to PNG/PDF/PS"),
        PackageInfo(name="wand", purpose="ImageMagick integration"),
        PackageInfo(name="svglib", purpose="SVG parsing"),
        PackageInfo(name="reportlab", purpose="PDF generation"),
        PackageInfo(name="drawSvg", purpose="SVG drawing")
    ],
    
    "ML (Optional)": [
        PackageInfo(
            name="xgboost",
            purpose="Gradient boosting",
            optional=True,
            dependencies=["numpy", "scipy"]
        ),
        PackageInfo(
            name="lightgbm",
            purpose="Gradient boosting framework",
            optional=True,
            dependencies=["numpy", "scipy"]
        ),
        PackageInfo(
            name="keras",
            purpose="Neural networks",
            optional=True,
            dependencies=["numpy"]
        ),
        PackageInfo(
            name="torch",
            purpose="PyTorch ML framework",
            optional=True,
            system_dependencies={
                "windows": ["Visual C++ Redistributable"],
                "linux": ["build-essential", "cmake"]
            }
        ),
        PackageInfo(
            name="torchvision",
            purpose="Computer vision with PyTorch",
            optional=True,
            dependencies=["torch", "pillow"]
        ),
        PackageInfo(
            name="torchaudio",
            purpose="Audio processing with PyTorch",
            optional=True,
            dependencies=["torch"]
        )
    ],
    
    "Terminal UX & Art": [
        PackageInfo(name="inquirer", purpose="Interactive command line prompts"),
        PackageInfo(name="PyInquirer", purpose="Interactive command line prompts"),
        PackageInfo(name="halo", purpose="Terminal spinners"),
        PackageInfo(name="yaspin", purpose="Terminal spinners"),
        PackageInfo(name="blessed", purpose="Terminal formatting"),
        PackageInfo(name="art", purpose="ASCII art"),
        PackageInfo(name="termimage", purpose="Terminal image display"),
        PackageInfo(name="click-option-group", purpose="Option groups for Click", dependencies=["click"])
    ],
    
    "Dev Tools": [
        PackageInfo(name="ipython", purpose="Enhanced interactive Python shell"),
        PackageInfo(name="black", purpose="Code formatter"),
        PackageInfo(name="isort", purpose="Import sorter"),
        PackageInfo(name="flake8", purpose="Linter"),
        PackageInfo(name="mypy", purpose="Static type checker"),
        PackageInfo(name="pylint", purpose="Linter"),
        PackageInfo(name="pdoc3", purpose="Documentation generator"),
        PackageInfo(name="mkdocs", purpose="Documentation site generator"),
        PackageInfo(name="mkdocstrings", purpose="Auto-documentation for MkDocs", dependencies=["mkdocs"]),
        PackageInfo(name="memory_profiler", purpose="Memory profiling", dependencies=["psutil"]),
        PackageInfo(name="jupyterlab", purpose="Jupyter notebook environment"),
        PackageInfo(name="notebook", purpose="Jupyter notebook"),
        PackageInfo(name="streamlit", purpose="Data app framework")
    ]
}


# ====== SYSTEM UTILITIES ======
class SystemUtility:
    """Utilities for system management and information."""
    
    @staticmethod
    def get_pip_path() -> Path:
        """Get path to pip executable."""
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # Inside a virtual environment
            if platform.system() == "Windows":
                return Path(sys.prefix) / "Scripts" / "pip.exe"
            else:
                return Path(sys.prefix) / "bin" / "pip"
        else:
            # System Python
            return Path(sys.executable).parent / ("pip.exe" if platform.system() == "Windows" else "pip")
    
    @staticmethod
    def get_python_path() -> Path:
        """Get path to Python executable."""
        return Path(sys.executable)
    
    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """Get detailed platform information."""
        info = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "name": platform.node(),
            "system_path": str(Path(sys.executable)),
            "pip_path": str(SystemUtility.get_pip_path()),
            "in_venv": str(hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)),
            "cpu_count": str(os.cpu_count()),
            "ram": str(SystemUtility.get_memory_info()),
            "disk_space": str(SystemUtility.get_disk_space())
        }
        return info
    
    @staticmethod
    def get_memory_info() -> str:
        """Get system memory information."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return f"{vm.total // (1024**3)} GB (Available: {vm.available // (1024**3)} GB)"
        except ImportError:
            return "Unknown (psutil not installed)"
    
    @staticmethod
    def get_disk_space() -> str:
        """Get disk space information."""
        try:
            total, used, free = shutil.disk_usage("/")
            return f"Total: {total // (1024**3)} GB, Free: {free // (1024**3)} GB"
        except Exception:
            return "Unknown"
    
    @staticmethod
    def is_admin() -> bool:
        """Check if running with admin/root privileges."""
        try:
            if platform.system() == "Windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except:
            return False
    
    @staticmethod
    def check_network_connectivity() -> bool:
        """Check if we have internet connectivity."""
        try:
            # Try to connect to PyPI
            urllib.request.urlopen("https://pypi.org", timeout=Config.NETWORK_TIMEOUT)
            return True
        except:
            try:
                # Fallback to Google
                urllib.request.urlopen("https://google.com", timeout=Config.NETWORK_TIMEOUT)
                return True
            except:
                return False
    
    @staticmethod
    def get_required_system_dependencies() -> Dict[str, List[str]]:
        """Get required system dependencies based on packages to install."""
        system = platform.system().lower()
        
        # Collect all system dependencies
        dependencies = {}
        for category, pkgs in PACKAGES.items():
            for pkg in pkgs:
                if system in pkg.system_dependencies:
                    for dep in pkg.system_dependencies[system]:
                        dependencies.setdefault(pkg.name, []).append(dep)
        
        return dependencies


# ====== PACKAGE MANAGEMENT ======
class PackageManager:
    """Manages package installation and verification."""
    
    def __init__(self):
        self.results: Dict[str, InstallResult] = {}
        self.dependency_graph = self._build_dependency_graph()
        self.progress = ProgressBar(self._count_all_packages())
        self.state = self._load_state()
        
    def _count_all_packages(self) -> int:
        """Count total number of packages."""
        count = 0
        for category, pkgs in PACKAGES.items():
            count += len(pkgs)
        return count
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build a dependency graph of all packages."""
        graph = {}
        
        # Add all packages first
        for category, pkgs in PACKAGES.items():
            for pkg in pkgs:
                graph[pkg.name] = set()
        
        # Add dependencies
        for category, pkgs in PACKAGES.items():
            for pkg in pkgs:
                for dep in pkg.dependencies:
                    if dep in graph:
                        graph[dep].add(pkg.name)
        
        return graph
    
    def _load_state(self) -> Dict[str, Any]:
        """Load saved installation state if it exists."""
        if Config.INSTALL_STATE_FILE.exists():
            try:
                with Config.INSTALL_STATE_FILE.open("r") as f:
                    return json.load(f)
            except:
                pass
        
        # Default state
        return {
            "started_at": time.time(),
            "completed": [],
            "failed": [],
            "skipped": [],
            "results": {}
        }
    
    def _save_state(self) -> None:
        """Save current installation state."""
        # Update state with current results
        self.state["updated_at"] = time.time()
        self.state["results"] = {
            pkg: {
                "status": result.status.value,
                "version": result.version,
                "message": result.message,
                "timestamp": result.timestamp
            }
            for pkg, result in self.results.items()
        }
        
        # Write to file
        with Config.INSTALL_STATE_FILE.open("w") as f:
            json.dump(self.state, f, indent=2)
    
    def get_package_info(self, name: str) -> Optional[PackageInfo]:
        """Get package info by name."""
        for category, pkgs in PACKAGES.items():
            for pkg in pkgs:
                if pkg.name == name:
                    return pkg
        return None
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages and versions."""
        packages = {}
        
        try:
            # Get list of installed packages
            result = subprocess.run(
                [str(SystemUtility.get_python_path()), "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                pkg_list = json.loads(result.stdout)
                for pkg in pkg_list:
                    packages[pkg["name"].lower()] = pkg["version"]
        except Exception as e:
            Logger.error(f"Failed to get installed packages: {str(e)}")
        
        return packages
    
    def get_topological_order(self) -> List[str]:
        """Get packages in topological order (dependencies first)."""
        # Kahn's algorithm for topological sorting
        in_degree = {node: 0 for node in self.dependency_graph}
        for node in self.dependency_graph:
            for dependent in self.dependency_graph[node]:
                in_degree[dependent] += 1
        
        # Priority queue (nodes with no incoming edges)
        package_priority = {}
        for category, pkgs in PACKAGES.items():
            for pkg in pkgs:
                package_priority[pkg.name] = pkg.priority
        
        # Sort by in-degree (0 first) and then by priority (higher first)
        queue = sorted(
            [node for node in in_degree if in_degree[node] == 0],
            key=lambda x: (-in_degree[x], -package_priority.get(x, 0))
        )
        
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for dependent in self.dependency_graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
            
            # Re-sort queue by priority
            queue.sort(key=lambda x: -package_priority.get(x, 0))
        
        # Check for cycles
        if len(result) != len(self.dependency_graph):
            Logger.warning("Dependency graph contains cycles!")
        
        return result
    
    def get_category_order(self) -> List[Tuple[str, List[PackageInfo]]]:
        """Get categories with their packages in optimized order."""
        installed = self.get_installed_packages()
        topological_order = self.get_topological_order()
        
        # Track the order of remaining packages
        remaining_order = {pkg: i for i, pkg in enumerate(topological_order)}
        
        # Sort each category by dependency order
        sorted_categories = []
        for category, pkgs in PACKAGES.items():
            # Sort packages within category by topological order
            sorted_pkgs = sorted(
                pkgs,
                key=lambda x: (
                    # First install already installed packages (for upgrading)
                    0 if x.name.lower() in installed else 1,
                    # Then by dependency order
                    remaining_order.get(x.name, 999999),
                    # Then by priority
                    -x.priority
                )
            )
            sorted_categories.append((category, sorted_pkgs))
        
        return sorted_categories
    
    def get_package_version(self, pkg_name: str) -> Optional[str]:
        """Get installed version of a package."""
        try:
            result = subprocess.run(
                [str(SystemUtility.get_python_path()), "-m", "pip", "show", pkg_name],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        
        return None
    
    def verify_package_import(self, pkg_name: str) -> bool:
        """Verify that a package can be imported."""
        if not Config.VERIFY_IMPORTS:
            return True
            
        # Some packages have different import names
        import_name_map = {
            "opencv-python-headless": "cv2",
            "opencv-contrib-python-headless": "cv2",
            "pillow": "PIL",
            "scikit-learn": "sklearn",
            "scikit-image": "skimage",
            "beautifulsoup4": "bs4",
            "python-dotenv": "dotenv",
        }
        
        # Get import name
        import_name = import_name_map.get(pkg_name, pkg_name)
        
        # Try to import
        try:
            # Use subprocess to avoid affecting current process
            import_cmd = f"import {import_name}"
            result = subprocess.run(
                [str(SystemUtility.get_python_path()), "-c", import_cmd],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            Logger.debug(f"Import verification failed for {pkg_name}: {str(e)}")
            return False

    def install_package(self, pkg_info: PackageInfo) -> InstallResult:
        """Install a package and return the result."""
        start_time = time.time()
        cmd = [str(SystemUtility.get_pip_path()), "install", "--upgrade"]
        
        # Add pip options
        if Config.PIP_INDEX_URL:
            cmd.extend(["--index-url", Config.PIP_INDEX_URL])
        if Config.PIP_EXTRA_INDEX_URL:
            cmd.extend(["--extra-index-url", Config.PIP_EXTRA_INDEX_URL])
        if Config.FORCE_REINSTALL:
            cmd.append("--force-reinstall")
        
        # Add package spec
        cmd.append(pkg_info.get_install_spec())
        
        # Log installation attempt
        Logger.debug(f"Running: {' '.join(cmd)}")
        
        # Run pip install
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=Config.TIMEOUT,
                check=False
            )
            
            # Process result
            if process.returncode == 0:
                version = self.get_package_version(pkg_info.name)
                if self.verify_package_import(pkg_info.name):
                    return InstallResult(
                        package=pkg_info.name,
                        status=PackageStatus.SUCCESS,
                        version=version,
                        message="Installed successfully",
                        duration=time.time() - start_time,
                        stdout=process.stdout,
                        stderr=process.stderr
                    )
                else:
                    return InstallResult(
                        package=pkg_info.name,
                        status=PackageStatus.FAILED,
                        version=version,
                        message="Installation succeeded but import verification failed",
                        duration=time.time() - start_time,
                        stdout=process.stdout,
                        stderr=process.stderr
                    )
            else:
                return InstallResult(
                    package=pkg_info.name,
                    status=PackageStatus.FAILED,
                    message=f"Installation failed with code {process.returncode}",
                    duration=time.time() - start_time,
                    stdout=process.stdout,
                    stderr=process.stderr
                )
        except subprocess.TimeoutExpired:
            return InstallResult(
                package=pkg_info.name,
                status=PackageStatus.FAILED,
                message=f"Installation timed out after {Config.TIMEOUT} seconds",
                duration=Config.TIMEOUT
            )
        except Exception as e:
            return InstallResult(
                package=pkg_info.name,
                status=PackageStatus.FAILED,
                message=f"Installation failed: {str(e)}",
                duration=time.time() - start_time
            )
            
    def install_all_packages(self) -> None:
        """Install all packages in the optimal order."""
        # Print system information
        sysinfo = SystemUtility.get_platform_info()
        Logger.info(f"System: {sysinfo['system']} {sysinfo['release']} ({sysinfo['architecture']})")
        Logger.info(f"Python: {sysinfo['python_version']} ({sysinfo['python_implementation']})")
        Logger.info(f"In virtual environment: {sysinfo['in_venv']}")
        
        # Check internet connectivity
        if not SystemUtility.check_network_connectivity():
            Logger.critical("No internet connectivity detected! Cannot install packages.")
            return
            
        # Get already installed packages
        installed_packages = self.get_installed_packages()
        Logger.info(f"Found {len(installed_packages)} already installed packages")
        
        # Get categories in order
        categories = self.get_category_order()
        total_packages = sum(len(pkgs) for _, pkgs in categories)
        
        Logger.info(f"Starting installation of {total_packages} packages in {len(categories)} categories")
        
        # Initialize progress bar
        self.progress = ProgressBar(total_packages)
        completed_count = 0
        
        # Process each category
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            for category, packages in categories:
                if not packages:
                    continue
                    
                # Log category
                Logger.info(f"\n{Colors.BOLD}Installing {len(packages)} packages from {category}{Colors.ENDC}")
                
                # Submit all packages in this category
                future_to_package = {}
                for pkg in packages:
                    # Skip optional packages unless explicitly requested
                    if pkg.optional and not Config.INSTALL_DEPS_FIRST:
                        self.results[pkg.name] = InstallResult(
                            package=pkg.name,
                            status=PackageStatus.SKIPPED,
                            message="Optional package skipped"
                        )
                        self.state["skipped"].append(pkg.name)
                        completed_count += 1
                        self.progress.update(completed_count)
                        continue
                        
                    # Check if already installed and up to date
                    current_version = self.get_package_version(pkg.name)
                    if current_version and not Config.FORCE_REINSTALL:
                        if pkg.min_version and current_version >= pkg.min_version:
                            self.results[pkg.name] = InstallResult(
                                package=pkg.name,
                                status=PackageStatus.SKIPPED,
                                version=current_version,
                                message="Already installed and up to date"
                            )
                            self.state["completed"].append(pkg.name)
                            completed_count += 1
                            self.progress.update(completed_count)
                            continue
                    
                    # Submit installation job
                    future = executor.submit(self.install_package, pkg)
                    future_to_package[future] = pkg
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_package):
                    pkg = future_to_package[future]
                    try:
                        result = future.result()
                        self.results[pkg.name] = result
                        
                        # Update state
                        if result.status == PackageStatus.SUCCESS:
                            self.state["completed"].append(pkg.name)
                            Logger.success(f"Installed {pkg.name} {result.version or ''}")
                        elif result.status == PackageStatus.FAILED:
                            self.state["failed"].append(pkg.name)
                            Logger.error(f"Failed to install {pkg.name}: {result.message}")
                        elif result.status == PackageStatus.SKIPPED:
                            self.state["skipped"].append(pkg.name)
                            Logger.info(f"Skipped {pkg.name}: {result.message}")
                    except Exception as e:
                        Logger.error(f"Error installing {pkg.name}: {str(e)}")
                        self.results[pkg.name] = InstallResult(
                            package=pkg.name,
                            status=PackageStatus.FAILED,
                            message=f"Unexpected error: {str(e)}"
                        )
                        self.state["failed"].append(pkg.name)
                    
                    # Update progress
                    completed_count += 1
                    self.progress.update(completed_count)
                    
                    # Save state after each package
                    self._save_state()
        
        # Finish progress bar
        self.progress.finish()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print installation summary."""
        # Count results by status
        counts = {status: 0 for status in PackageStatus}
        for pkg, result in self.results.items():
            counts[result.status] += 1
            
        # Print totals
        Logger.info(f"\n{Colors.BOLD}Installation Summary{Colors.ENDC}")
        Logger.info(f"Total packages: {len(self.results)}")
        Logger.info(f"  {Colors.GREEN}Installed: {counts[PackageStatus.SUCCESS]}{Colors.ENDC}")
        Logger.info(f"  {Colors.YELLOW}Skipped: {counts[PackageStatus.SKIPPED]}{Colors.ENDC}")
        Logger.info(f"  {Colors.RED}Failed: {counts[PackageStatus.FAILED]}{Colors.ENDC}")
        
        # Print failed packages if any
        if counts[PackageStatus.FAILED] > 0:
            Logger.info(f"\n{Colors.BOLD}Failed Packages:{Colors.ENDC}")
            for pkg, result in self.results.items():
                if result.status == PackageStatus.FAILED:
                    Logger.info(f"  {Colors.RED}{pkg}: {result.message}{Colors.ENDC}")
                    
        # Print category stats
        if Config.SHOW_CATEGORY_STATS:
            Logger.info(f"\n{Colors.BOLD}Category Statistics:{Colors.ENDC}")
            for category, packages in PACKAGES.items():
                success = 0
                failed = 0
                skipped = 0
                
                for pkg in packages:
                    if pkg.name in self.results:
                        status = self.results[pkg.name].status
                        if status == PackageStatus.SUCCESS:
                            success += 1
                        elif status == PackageStatus.FAILED:
                            failed += 1
                        elif status == PackageStatus.SKIPPED:
                            skipped += 1
                
                total = len(packages)
                if total > 0:
                    success_rate = (success / total) * 100
                    color = Colors.GREEN if success_rate > 90 else Colors.YELLOW if success_rate > 50 else Colors.RED
                    Logger.info(f"  {category}: {color}{success}/{total} ({success_rate:.1f}%){Colors.ENDC}")


# ====== COMMAND LINE INTERFACE ======
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BEASTIQUE COSMIC INSTALLER - Advanced Python package installer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--categories",
        type=str,
        help="Comma-separated list of categories to install (default: all)"
    )
    
    parser.add_argument(
        "-p", "--packages",
        type=str,
        help="Comma-separated list of specific packages to install"
    )
    
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force reinstallation of packages even if already installed"
    )
    
    parser.add_argument(
        "--verify-imports",
        action="store_true",
        default=Config.VERIFY_IMPORTS,
        help="Verify that packages can be imported after installation"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=Config.MAX_WORKERS,
        help="Maximum number of parallel installation workers"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=Config.TIMEOUT,
        help="Timeout in seconds for each package installation"
    )
    
    parser.add_argument(
        "--index-url",
        type=str,
        default=Config.PIP_INDEX_URL,
        help="Custom PyPI index URL"
    )
    
    parser.add_argument(
        "--extra-index-url",
        type=str,
        default=Config.PIP_EXTRA_INDEX_URL,
        help="Extra PyPI index URL"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Silent mode (no console output)"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available package categories and exit"
    )
    
    parser.add_argument(
        "--list-packages",
        action="store_true",
        help="List all available packages and exit"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check which packages would be installed but don't install them"
    )
    
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional packages"
    )
    
    return parser.parse_args()


def apply_arguments(args: argparse.Namespace) -> None:
    """Apply command-line arguments to configuration."""
    if args.force_reinstall:
        Config.FORCE_REINSTALL = True
        
    if args.verify_imports is not None:
        Config.VERIFY_IMPORTS = args.verify_imports
        
    if args.workers:
        Config.MAX_WORKERS = args.workers
        
    if args.timeout:
        Config.TIMEOUT = args.timeout
        
    if args.index_url:
        Config.PIP_INDEX_URL = args.index_url
        
    if args.extra_index_url:
        Config.PIP_EXTRA_INDEX_URL = args.extra_index_url
        
    if args.no_progress:
        Config.PROGRESS_BAR = False
        
    if args.debug:
        Config.DEBUG_MODE = True
        Logger.current_level = 0  # Debug level
        
    if args.silent:
        Config.SILENT_MODE = True


def list_categories() -> None:
    """List available package categories."""
    print(f"\n{Colors.BOLD}Available Package Categories:{Colors.ENDC}")
    for category, packages in PACKAGES.items():
        print(f"  {Colors.BLUE}{category}{Colors.ENDC}: {len(packages)} packages")


def list_packages() -> None:
    """List all available packages with details."""
    print(f"\n{Colors.BOLD}All Available Packages:{Colors.ENDC}")
    for category, packages in PACKAGES.items():
        print(f"\n{Colors.BLUE}{category}{Colors.ENDC}:")
        for pkg in packages:
            status = f"{Colors.YELLOW}[Optional]{Colors.ENDC}" if pkg.optional else ""
            version_req = ""
            if pkg.min_version and pkg.max_version:
                version_req = f">={pkg.min_version}, <={pkg.max_version}"
            elif pkg.min_version:
                version_req = f">={pkg.min_version}"
            elif pkg.max_version:
                version_req = f"<={pkg.max_version}"
                
            if version_req:
                version_req = f" ({version_req})"
                
            print(f"  {Colors.CYAN}{pkg.name}{Colors.ENDC}{version_req} {status}")
            if pkg.purpose:
                print(f"    {Colors.GRAY}{pkg.purpose}{Colors.ENDC}")
            if pkg.dependencies:
                print(f"    {Colors.GRAY}Dependencies: {', '.join(pkg.dependencies)}{Colors.ENDC}")


def main() -> None:
    """Main entry point."""
    # Parse command-line arguments
    args = parse_arguments()
    apply_arguments(args)
    
    # Setup logging
    if Config.LOG_FILE.exists():
        Config.LOG_FILE.unlink()  # Clear existing log
        
    # Print welcome message
    if not Config.SILENT_MODE:
        print(f"\n{Colors.BOLD}{Colors.BLUE}BEASTIQUE COSMIC INSTALLER{Colors.ENDC}")
        print(f"{Colors.GRAY}A hyper-optimized Python package installer{Colors.ENDC}\n")
    
    # Handle special commands
    if args.list_categories:
        list_categories()
        return
        
    if args.list_packages:
        list_packages()
        return
    
    # Initialize package manager
    manager = PackageManager()
    
    # Filter packages if requested
    if args.categories or args.packages:
        filtered_packages = {}
        
        if args.categories:
            categories = [c.strip() for c in args.categories.split(",")]
            for category in categories:
                if category in PACKAGES:
                    filtered_packages[category] = PACKAGES[category]
                else:
                    Logger.warning(f"Category not found: {category}")
        
        if args.packages:
            package_names = [p.strip() for p in args.packages.split(",")]
            custom_category = "Selected Packages"
            filtered_packages[custom_category] = []
            
            for name in package_names:
                found = False
                for category, pkgs in PACKAGES.items():
                    for pkg in pkgs:
                        if pkg.name == name:
                            filtered_packages[custom_category].append(pkg)
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    Logger.warning(f"Package not found: {name}")
        
        # Replace original packages with filtered ones
        global PACKAGES
        PACKAGES = filtered_packages
    
    # Start installation process
    if not args.check_only:
        manager.install_all_packages()
    else:
        # Just print what would be installed
        print(f"\n{Colors.BOLD}Packages that would be installed:{Colors.ENDC}")
        for category, packages in PACKAGES.items():
            print(f"\n{Colors.BLUE}{category}{Colors.ENDC}:")
            for pkg in packages:
                print(f"  {Colors.CYAN}{pkg.name}{Colors.ENDC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        if Config.DEBUG_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)

# ︻デ═—··· 🎯
