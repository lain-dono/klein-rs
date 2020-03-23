/// Project a point onto a line.
#[inline]
pub fn project_point_to_line(a: Point, b: Line) -> Point {
    (a | b) ^ b
}

/// Project a point onto a plane.
#[inline]
pub fn project_point_to_plane(a: Point, b: Plane) -> Point {
    (a | b) ^ b
}

/// Project a line onto a plane.
#[inline]
pub fn project_line_to_plane(a: Line, b: Plane) -> Line {
    (a | b) ^ b
}

/// Project a plane onto a point.
///
/// Given a plane `p` and point `P`, produces the
/// plane through `P` that is parallel to `p`.
///
/// Intuitively, the point is represented dually in terms of a _pencil of planes_
/// that converge on the point itself. When we compute `p | P`, this
/// selects the line perpendicular to `p` through `P`. Subsequently, taking the
/// inner product with $`P`$ again selects the plane from the plane pencil of `P`
/// _least like_ that line.
#[inline]
pub fn project_plane_to_point(a: Plane, b: Point) -> Plane {
    (a | b) | b
}

/// Project a line onto a point. Given a line $`\ell`$ and point $`P`$, produces the
/// line through $`P`$ that is parallel to $`\ell`$.
#[inline]
pub fn project_line_to_point(a: Line, b: Point) -> Line {
    (a | b) | b
}

/// Project a plane onto a line. Given a plane $`p`$ and line $`\ell`$, produces the
/// plane through $`\ell`$ that is parallel to $`p`$ if $`p \parallel \ell`$.
///
/// If $`p \nparallel \ell`$, the result will be the plane $`p'`$ containing $`\ell`$
/// that maximizes $`p \cdot p'`$ (that is, $`p'`$ is as parallel to $`p`$ as possible).
#[inline]
pub fn project_plane_to_line(a: Plane, b: Line) -> Plane {
    (a | b) | b
}
