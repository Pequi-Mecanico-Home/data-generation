from blender_sdg.core.interfaces.blender.object import BlenderElement
from blender_sdg.core.interfaces.blender.scene import BlenderScene
from typing import List, Tuple, Optional
import numpy as np
import bpy

import mathutils


def create_bounding_box(
    scene: BlenderScene,
    camera: BlenderElement,
    element: BlenderElement,
    relative: bool = True,
    resolution: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Create bounding boxes for the elements in the scene.

    Parameters:
    -----------
    scene: BlenderScene
        The scene to create bounding boxes for.
    camera: BlenderElement
        The camera to create bounding boxes for.
    element: BlenderElement
        The element to create bounding boxes for.
    relative: bool
        Whether to return the bounding box in relative coordinates.
    resolution: Optional[Tuple[int, int]]
        The resolution to normalize the bounding box to.

    Returns:
    --------
    Optional[np.ndarray]
        The bounding boxes for the elements in the scene, or None if not in view.
    """
    # Prepare transformation matrix from camera to object
    inverse_matrix = camera.get_matrix(inverse=True, normalized=True)
    mesh = element.get_mesh(preserve_all_data_layers=True)

    # Apply transformation matrix to mesh, where the mesh is
    # first transformed to the object's coordinate space,
    mesh.transform(element.get_matrix())
    # then to the camera's coordinate space
    mesh.transform(inverse_matrix)

    # Prepare camera frame 3D coordinates
    frame = [-v for v in camera.object.data.view_frame(scene=scene.blender_scene)[:3]]

    # Then get the object's mesh vertices 2D coordinates
    # in the camera's coordinate space
    lx, ly = calculate_normalized_coordinates(mesh, frame)

    # Finally, compute the bounding box from the 2D coordinates
    return compute_bounding_box(lx, ly, relative=relative, resolution=resolution)


def calculate_normalized_coordinates(
    mesh: bpy.types.Mesh, frame: List[mathutils.Vector]
) -> Tuple[List[float], List[float]]:
    """Calculate the normalized coordinates of the mesh vertices.

    Parameters:
    -----------
    mesh: bpy.types.Mesh
        The mesh to calculate coordinates for.
    frame: List[mathutils.Vector]
        The camera frame to use for calculations.

    Returns:
    --------
    Tuple[List[float], List[float]]
        The normalized x and y coordinates of the mesh vertices.
    """
    lx = []
    ly = []

    for v in mesh.vertices:
        co_local = v.co
        z = -co_local.z

        if z <= 0.0:
            continue
        else:
            frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    return lx, ly


def compute_bounding_box(
    lx: List[float],
    ly: List[float],
    relative: bool = True,
    resolution: Tuple[int, int] = (256, 256),
) -> Optional[np.ndarray]:
    """Compute the bounding box from the normalized coordinates.

    Parameters:
    -----------
    lx: List[float]
        The list of normalized x coordinates.
    ly: List[float]
        The list of normalized y coordinates.
    relative: bool
        Whether to return the bounding box in relative coordinates.
    resolution: Tuple[int, int]
        The resolution to normalize the bounding box to.

    Returns:
    --------
    Optional[np.ndarray]
        The computed bounding box, or None if not in view.
        The format returned is (top_x, top_y, width, height)
    """
    if not lx or not ly:
        return None

    # Calculate the bounding box limits
    top_x, bottom_x = np.clip([min(lx), max(lx)], 0.0, 1.0)
    top_y, bottom_y = np.clip([min(ly), max(ly)], 0.0, 1.0)

    # Check if the bounding box is valid
    if top_x == bottom_x or top_y == bottom_y:
        return None

    # Convert to relative coordinates if required
    if relative:
        top_x, top_y, bottom_x, bottom_y = make_bounding_box_relative(
            top_x, top_y, bottom_x, bottom_y, resolution
        )

    # Calculate width and height
    width = bottom_x - top_x
    height = bottom_y - top_y

    # Return the bounding box as a numpy array
    return np.array([top_x or 1e-6, top_y or 1e-6, width, height])


def make_bounding_box_relative(
    top_x: float,
    top_y: float,
    bottom_x: float,
    bottom_y: float,
    resolution: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Make the bounding box relative to the resolution.

    Parameters:
    -----------
    top_x: float
        The top x coordinate of the bounding box.
    top_y: float
        The top y coordinate of the bounding box.
    bottom_x: float
        The bottom x coordinate of the bounding box.
    bottom_y: float
        The bottom y coordinate of the bounding box.
    resolution: Tuple[int, int]
        The resolution to normalize the bounding box to.

    Returns:
    --------
    Tuple[float, float, float, float]
        The bounding box in relative coordinates.

    Raises:
    -------
    ValueError
        If the resolution is not provided.
    """
    if resolution is None:
        raise ValueError("Resolution must be provided if relative is True")

    return (
        top_x * resolution[0],
        top_y * resolution[1],
        bottom_x * resolution[0],
        bottom_y * resolution[1],
    )
