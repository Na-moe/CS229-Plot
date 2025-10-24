from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


def config_mpl():
    plt.rcParams["backend"] = "svg"
    mpl.style.use(["ieee"])  # type: ignore
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["lines.linewidth"] = 1
    mpl.rc(
        group="font",
        **{
            "family": ["serif", "sans-serif"],
            "serif": ["Times New Roman", "LXGW Bright GB", "SimSun"],
            "sans-serif": ["LXGW Bright GB", "Arial"],
            "weight": "normal",
            "size": 10,
        },
    )
    plt.rcParams["pdf.fonttype"] = 42


def get_save_path(file_path: Path) -> Path:
    # Set Save Path
    root = file_path.parent.parent.parent

    chapter = file_path.parent
    file_name = file_path.stem

    save_dir: Path = root / "figs" / chapter.name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path: Path = save_dir / f"{file_name}.svg"
    return save_path
