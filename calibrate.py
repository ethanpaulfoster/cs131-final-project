from typing import Tuple

import numpy as np

def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == float

    # Intersection point between lines
    out = np.zeros(2)

    # YOUR CODE HERE
    lines = np.zeros((2, 2))
    #each line has length 2, for slope and y intercept
    
    lines[0][0] = (a_0[1]-a_1[1])/(a_0[0]-a_1[0])
    lines[0][1] = (a_0[1] - lines[0][0]*a_0[0])
    
    lines[1][0] = (b_0[1]-b_1[1])/(b_0[0]-b_1[0])
    lines[1][1] = (b_0[1] - lines[1][0]*b_0[0])
    
    out[0] = (lines[1][1] - lines[0][1])/(lines[0][0] - lines[1][0])
    out[1] = lines[0][0] * out[0] + lines[0][1]
    # END YOUR CODE

    assert out.shape == (2,)
    assert out.dtype == float
   
    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # YOUR CODE HERE
    lines = np.zeros((2, 2))
    #each line has length 2, for slope and y intercept
    
    #lines will be perpendicular to triangle edges
    lines[0][0] = -(v0[0] - v1[0])/(v0[1] - v1[1])
    lines[1][0] = -(v1[0] - v2[0])/(v1[1] - v2[1])
    
    lines[0][1] = v2[1] - lines[0][0] * v2[0]
    lines[1][1] = v0[1] - lines[1][0] * v0[0]
    
    optical_center[0] = (lines[1][1] - lines[0][1])/(lines[0][0] - lines[1][0])
    optical_center[1] = lines[0][0] * optical_center[0] + lines[0][1]

    # END YOUR CODE
    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    # YOUR CODE HERE
    f_squared = (-1 * v0[0] * v1[0]) + (v0[0] * optical_center[0]) + (v1[0] * optical_center[0]) - (optical_center[0] ** 2) - (v0[1] * v1[1]) + (v0[1] * optical_center[1]) + (v1[1] * optical_center[1]) - (optical_center[1] ** 2)
    f = np.sqrt(f_squared)
    # END YOUR CODE

    return float(f)