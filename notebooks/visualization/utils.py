import io
import os
import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from IPython.display import FileLink, Video, display
from ripser import ripser

# get repo root and add to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# try to use data manager
# otherwise use directory-based loading for local files (CPU tensors only)
try:
    from tda_utils import TDADataManager, tda_manager
except ImportError:
    TDADataManager = None
    tda_manager = None



class CPU_Unpickler(pickle.Unpickler):
    """ unpickler that maps CUDA tensors to CPU during loading"""


    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: self.load_tensor_from_bytes(b)
        return super().find_class(module, name)

    def load_tensor_from_bytes(self, b):
        import torch

        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


def betti_at_t(intervals, t):
    """ returns betti number at time t for a given set of intervals"""

    if intervals is None or len(intervals) == 0:
        return 0
    births = intervals[:, 0]
    deaths = intervals[:, 1]
    alive = (births <= t) & ((deaths > t) | np.isinf(deaths))
    return int(np.count_nonzero(alive))


def load_embeddings_from_directory(directory_path):
    """ loads embeddings from all pickle files in a directory - works with CPU and GPU tensors"""

    directory_path = Path(directory_path)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    embedding_dict = {}
    pickle_files = list(directory_path.glob("*.pkl")) + list(directory_path.glob("*.pickle"))
    if not pickle_files:
        raise ValueError(f"No pickle files found in directory: {directory_path}")

    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False

    for pkl_file in pickle_files:
        try:
            with open(pkl_file, "rb") as f:
                loaded_data = CPU_Unpickler(f).load() if has_torch else pickle.load(f)
        except Exception as exc:
            print(f"Skipping {pkl_file.name}: {exc}")
            continue

        file_stem = pkl_file.stem

        if isinstance(loaded_data, tuple) and len(loaded_data) >= 1:
            loaded_data = loaded_data[0]

        if not isinstance(loaded_data, dict):
            print(f"Skipping {pkl_file.name}: unsupported format {type(loaded_data)}")
            continue

        session_names = list(loaded_data.keys())
        for session_name in session_names:
            session_payload = loaded_data.get(session_name, {})
            if not isinstance(session_payload, dict) or "embedding" not in session_payload:
                continue

            embedding = session_payload["embedding"]
            if hasattr(embedding, "cpu"):
                embedding = embedding.cpu().numpy()
            elif hasattr(embedding, "numpy"):
                embedding = embedding.numpy()
            else:
                embedding = np.asarray(embedding)

            unique_name = f"{file_stem}_{session_name}" if len(session_names) > 1 else file_stem
            embedding_dict[unique_name] = {
                "embedding": embedding,
                "original_session": session_name,
                "filename": pkl_file.name,
            }

    if not embedding_dict:
        raise ValueError("No embeddings could be loaded from the directory.")

    return embedding_dict


def get_tda_data_manager(workspace_root=None, data_manager=None):
    """Return an initialized TDA data manager for this repository."""

    if data_manager is not None:
        return data_manager

    if tda_manager is not None:
        return tda_manager

    if TDADataManager is None:
        raise ImportError("Could not import TDADataManager from tda_utils.")

    resolved_root = Path(workspace_root).resolve() if workspace_root else REPO_ROOT
    return TDADataManager(workspace_root=resolved_root)


def list_repo_embeddings(workspace_root=None, data_manager=None, force_cpu=False):
    """List embedding filenames and available sessions from the repo data manager."""

    manager = get_tda_data_manager(workspace_root=workspace_root, data_manager=data_manager)
    embedding_files = sorted(manager.find_files("*.pkl", "cebra_examples"))

    embeddings = {}
    for embedding_path in embedding_files:
        session_dict, session_names = manager.load_embedding_data(embedding_path, force_cpu=force_cpu)
        embeddings[embedding_path.name] = {
            "path": embedding_path,
            "session_names": session_names,
            "session_count": len(session_names),
            "session_dict": session_dict,
        }

    return embeddings


def load_repo_embedding(
    embedding_filename,
    session_name=None,
    workspace_root=None,
    data_manager=None,
    force_cpu=True,
):
    """Load one embedding session from repository-managed sample data."""

    manager = get_tda_data_manager(workspace_root=workspace_root, data_manager=data_manager)
    embedding_files = sorted(manager.find_files("*.pkl", "cebra_examples"))

    if not embedding_files:
        raise ValueError("No repository embedding files were found in data/CEBRA_embedding_examples.")

    matches = [path for path in embedding_files if embedding_filename.lower() in path.name.lower()]
    if not matches:
        available_files = [path.name for path in embedding_files]
        raise ValueError(
            f"Embedding file '{embedding_filename}' not in repository data. "
            f"Available files: {available_files}"
        )

    embedding_path = matches[0]
    session_dict, session_names = manager.load_embedding_data(embedding_path, force_cpu=force_cpu)

    if session_name is None:
        session_name = session_names[0]
    else:
        matched_session = next((name for name in session_names if name.lower() == session_name.lower()), None)
        if matched_session is None:
            raise ValueError(
                f"Session '{session_name}' not found in {embedding_path.name}. "
                f"Available sessions: {session_names}"
            )
        session_name = matched_session

    embedding = session_dict[session_name]["embedding"]
    if hasattr(embedding, "cpu"):
        embedding = embedding.cpu().numpy()
    elif hasattr(embedding, "numpy"):
        embedding = embedding.numpy()
    else:
        embedding = np.asarray(embedding)

    return {
        "embedding": embedding,
        "session_name": session_name,
        "embedding_path": embedding_path,
        "session_names": session_names,
    }


def downsample_embedding(embedding, n_samples=None, sampling_method="random", random_seed=42):
    """ downsample embedding to n_samples using specified sampling method"""

    n_points = len(embedding)
    if n_samples is None:
        n_samples = min(500, max(50, int(0.1 * n_points)))
    if n_samples >= n_points:
        return embedding, list(range(n_points))

    rng = np.random.default_rng(random_seed)
    if sampling_method == "random":
        indices = rng.choice(n_points, size=n_samples, replace=False)
    elif sampling_method == "uniform":
        indices = np.linspace(0, n_points - 1, n_samples, dtype=int)
    elif sampling_method == "first":
        indices = np.arange(n_samples)
    else:
        raise ValueError("sampling_method must be 'random', 'uniform', or 'first'")

    indices = np.sort(indices)
    return embedding[indices], indices.tolist()


def increase_ball_epsilon(embedding, diameter):
    """ increases the 'balls' for the VR filtrations & draws connections between points with betti numbers at given epsilon"""

    epsilon = diameter / 2.0
    n_points = len(embedding)
    balls = {
        i: {"center": embedding[i], "radius": epsilon, "center_idx": i}
        for i in range(n_points)
    }

    # connect points with overlapping balls (distance <= diameter)
    connections = []
    for i in range(n_points):
        pi = embedding[i]
        for j in range(i + 1, n_points):
            pj = embedding[j]
            if np.linalg.norm(pi - pj) <= diameter:
                connections.append((i, j))

    dgms = ripser(embedding, maxdim=2, thresh=diameter)["dgms"]
    return {
        "balls": balls,
        "connections": connections,
        "epsilon": epsilon,
        "diameter": diameter,
        "betti0": betti_at_t(dgms[0], diameter) if len(dgms) > 0 else 0,
        "betti1": betti_at_t(dgms[1], diameter) if len(dgms) > 1 else 0,
        "betti2": betti_at_t(dgms[2], diameter) if len(dgms) > 2 else 0,
    }


def create_3d_sphere(ax, center, radius, color="lightblue", alpha=0.01):
    """ draws 3d spheres around points to vsiualize the VR filtration sweep"""

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.2, edgecolor="steelblue")


def prepare_embedding_for_viz(embedding):
    """ checks embedding dimensionality & adds zero z-axis if only 2D, or truncates to 3D if higher dimensionality"""

    if embedding.shape[1] == 2:
        return np.column_stack([embedding, np.zeros(len(embedding))])
    if embedding.shape[1] >= 3:
        return embedding[:, :3]
    raise ValueError("embedding must have at least 2 dimensions")


def render_filtration_frame(embedding_3d, diameter, show_spheres=True, elev=15, azim=45):
    """ renders a single frame of the VR filtration visualization at a given epsilon"""

    res = increase_ball_epsilon(embedding_3d, diameter)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = np.linspace(0, 1, len(embedding_3d))

    # plot points
    ax.scatter(
        embedding_3d[:, 0],
        embedding_3d[:, 1],
        embedding_3d[:, 2],
        c=colors,
        cmap="hsv",
        s=18,
        alpha=0.95,
        edgecolor="black",
        linewidth=0.35)

    if show_spheres:
        for ball in res["balls"].values():
            create_3d_sphere(ax, ball["center"], ball["radius"])

    # draw connections in 3D plot
    for i, j in res["connections"]:
        p1 = res["balls"][i]["center"]
        p2 = res["balls"][j]["center"]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            c="0.5",
            alpha=0.7,
            linewidth=0.9)

    # title w epsilon and betti number count
    ax.set_title(
        f"ε={diameter:.3f}\n"
        f"β₀={res['betti0']}  β₁={res['betti1']}  β₂={res['betti2']}",
        fontsize=16,
        pad=16)
    
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def create_filtration_mp4(embedding,output_path,diameter_range=(0.0, 0.5),n_frames=24,
                          fps=6, show_spheres=True, elev=15,azim=45):
    """ creates an MP4 video visualizing the VR filtration progression across a range of epsilon values"""

    embedding_3d = prepare_embedding_for_viz(embedding)
    diameters = np.linspace(diameter_range[0], diameter_range[1], n_frames)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    fig = plt.figure(figsize=(8, 8))

    try:
        with writer.saving(fig, str(output_path), dpi=120):
            for frame_idx, diameter in enumerate(diameters, start=1):
                fig.clf()
                frame_ax = fig.add_subplot(111, projection="3d")
                res = increase_ball_epsilon(embedding_3d, diameter)

                colors = np.linspace(0, 1, len(embedding_3d))

                # plot points
                frame_ax.scatter(
                    embedding_3d[:, 0],
                    embedding_3d[:, 1],
                    embedding_3d[:, 2],
                    c=colors,
                    cmap="hsv",
                    s=18,
                    alpha=0.95,
                    edgecolor="black",
                    linewidth=0.35)

                if show_spheres:
                    for ball in res["balls"].values():
                        create_3d_sphere(frame_ax, ball["center"], ball["radius"])

                for i, j in res["connections"]:
                    p1 = res["balls"][i]["center"]
                    p2 = res["balls"][j]["center"]
                    frame_ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        c="0.5",
                        alpha=0.7,
                        linewidth=0.9)

                frame_ax.set_title(
                    f"ε={diameter:.3f}\n"
                    f"β₀={res['betti0']}  β₁={res['betti1']}  β₂={res['betti2']}",
                    fontsize=16,
                    pad=16)
                
                frame_ax.set_box_aspect([1, 1, 1])
                frame_ax.view_init(elev=elev, azim=azim)
                frame_ax.set_axis_off()
                fig.tight_layout()
                writer.grab_frame()
                print(f"Frame {frame_idx}/{n_frames}")
    except FileNotFoundError as exc:
        plt.close(fig)
        raise RuntimeError(
            "ffmpeg is required to write MP4 files. Install ffmpeg then rerun the export cell."
        ) from exc
    finally:
        plt.close(fig)

    return output_path


def export_filtration_mp4(pickle_directory_path=None, mouse_name=None, output_name="filtration.mp4", n_samples=300, sampling_method="random",
                          random_seed=42, diameter_range=(0.0, 0.5), n_frames=24, fps=6, show_spheres=True,
                          embedding_filename=None, session_name=None, workspace_root=None, data_manager=None,
                          force_cpu=True):
    """ main workflow function to load embedding, downsample, create filtration video, and display video w download link"""

    embedding_dict = None
    embedding_source = None

    if embedding_filename is not None:
        loaded_embedding = load_repo_embedding(
            embedding_filename=embedding_filename,
            session_name=session_name,
            workspace_root=workspace_root,
            data_manager=data_manager,
            force_cpu=force_cpu,
        )
        original_embedding = loaded_embedding["embedding"]
        selected_name = loaded_embedding["session_name"]
        embedding_source = loaded_embedding["embedding_path"]
    else:
        if pickle_directory_path is None or mouse_name is None:
            raise ValueError(
                "use either `embedding_filename` for repository data or both "
                "`pickle_directory_path` and `mouse_name` for a directory-based export"
            )

        embedding_dict = load_embeddings_from_directory(pickle_directory_path)
        if mouse_name not in embedding_dict:
            raise ValueError(f"mouse '{mouse_name}' not found. available: {list(embedding_dict.keys())}")

        original_embedding = embedding_dict[mouse_name]["embedding"]
        selected_name = mouse_name
        embedding_source = Path(pickle_directory_path)

    downsampled_embedding, selected_indices = downsample_embedding(
        original_embedding,
        n_samples=n_samples,
        sampling_method=sampling_method,
        random_seed=random_seed)

    output_path = Path.cwd() / output_name
    video_path = create_filtration_mp4(
        downsampled_embedding,
        output_path=output_path,
        diameter_range=diameter_range,
        n_frames=n_frames,
        fps=fps,
        show_spheres=show_spheres)

    print(f"original embedding shape: {original_embedding.shape}")
    print(f"downsampled embedding shape: {downsampled_embedding.shape}")
    print(f"selected embedding: {selected_name}")
    print(f"embedding source: {embedding_source}")
    print(f"saved MP4 to: {video_path}")
    display(Video(str(video_path), embed=False))
    display(FileLink(str(video_path), result_html_prefix="download MP4: "))

    return {
        "embedding_dict": embedding_dict,
        "original_embedding": original_embedding,
        "downsampled_embedding": downsampled_embedding,
        "selected_indices": selected_indices,
        "selected_name": selected_name,
        "embedding_source": embedding_source,
        "video_path": video_path,
    }
