//! # Quick start
//!
//! Here's a simple snippet to get you started:
//! ```rust
//! use klein::{Motor, Point, Rotor, Translator};
//!
//! // Create a rotor representing a pi/2 rotation about the z-axis
//! // Normalization is done automatically
//! let r = Rotor::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0, 1.0);
//!
//! // Create a translator that represents a translation of 1 unit
//! // in the yz-direction. Normalization is done automatically.
//! let t = Translator::new(1.0, 0.0, 1.0, 1.0);
//!
//! // Create a motor that combines the action of the rotation and
//! // translation above.
//! let m: Motor = r * t;
//!
//! // Construct a point at position (1, 0, 0)
//! let p1 = Point::new(1.0, 0.0, 0.0);
//!
//! // Apply the motor to the point. This is equivalent to the conjugation
//! // operator m * p1 * m.reversed() where * is the geometric product.
//! let p2: Point = m.conj_point(p1);
//!
//! // We could have also written p2 = m * p1 * ~m but this will be slower
//! // because the call operator eliminates some redundant or cancelled
//! // computation.
//! // point p2 = m * p1 * ~m;
//!
//! // We can access the coordinates of p2 with p2.x(), p2.y(), p2.z(),
//! // and p2.w(), where p.2w() is the homogeneous coordinate (initialized
//! // to one). It is recommended to localize coordinate access in this way
//! // as it requires unpacking storage that may occupy an SSE register.
//!
//! // Rotors and motors can produce 4x4 transformation matrices suitable
//! // for upload to a shader or for interoperability with code expecting
//! // matrices as part of its interface. The matrix returned in this way
//! // is a column-major matrix
//! //mat4x4 m_matrix = m.as_mat4x4();
//! ```
//!
//! The spherical interpolation (aka slerp) employed to produce smooth incremental rotations/transformations
//! in the quaternion algebra is available in Klein using the exp and log functions as in the snippet below.
//!
//! ```rust
//! use klein::{Motor, Line};
//!
//! // Blend between two motors with a parameter t in the range [0, 1]
//! fn blend_motors(a: Motor, b: Motor, t: f32) -> Motor {
//!     // Starting from a, the motor needed to get to b is `b * a.reversed()`.
//!     // To perform this motion continuously, we can take the principal
//!     // branch of the logarithm of `b * a.reverse()`, and subdivide it before
//!     // re-exponentiating it to produce a motor again.
//!
//!     // In practice, this should be cached whenever possible.
//!     let step: Line = (b * a.reversed()).log();
//! 
//!     // exp(log(m)) = exp(t*log(m) + (1 - t)*log(m))
//!     // = exp(t*(log(m))) * exp((1 - t)*log(m))
//!     let step = step * t;
//! 
//!     // The exponential of the step here can be cached if the blend occurs
//!     // with fixed steps toward the final motor. Compose the interpolated
//!     // result with the start motor to produce the intermediate blended motor.
//!     step.exp() * a
//! }
//! ```
//!
//! ## Points
//!
//! A point is represented as the multivector
//! $`x\mathbf{e}_{032} + y\mathbf{e}_{013} + z\mathbf{e}_{021} + \mathbf{e}_{123}`$.
//!
//! The point has a trivector representation because it is
//! the fixed point of 3 planar reflections (each of which is a grade-1 multivector).
//! In practice, the coordinate mapping can be thought of as an implementation detail.
//!
//! # Projections
//!
//! Projections in Geometric Algebra take on a particularly simple form.
//! For two geometric entities $a$ and $b$, there are two cases to consider.
//! First, if the grade of $a$ is greater than the grade of $b$, the projection
//! of $a$ on $b$ is given by:
//!
//! $` \textit{proj}_b a = (a \cdot b) \wedge b `$
//!
//! The inner product can be thought of as the part of $b$ _least like_ $a$.
//! Using the meet operator on this part produces the part of $b$ _most like_
//! $a$. A simple sanity check is to consider the grades of the result. If the
//! grade of $b$ is less than the grade of $a$, we end up with an entity with
//! grade $a - b + b = a$ as expected.
//!
//! In the second case (the grade of $a$ is less than the grade of $b$), the
//! projection of $a$ on $b$ is given by:
//!
//! $` \textit{proj}_b a = (a \cdot b) \cdot b `$
//!
//! It can be verified that as in the first case, the grade of the result is the
//! same as the grade of $a$. As this projection occurs in the opposite sense
//! from what one may have seen before, additional clarification is provided below.
//!
//! ### Poincaré Dual (dual)
//!
//! The Poincaré Dual of an element is the "subspace complement" of the
//! argument with respect to the pseudoscalar in the exterior algebra. In
//! practice, it is a relabeling of the coordinates to their
//! dual-coordinates and is used most often to implement a "join" operation
//! in terms of the exterior product of the duals of each operand.
//!
//! Ex: The dual of the point $`\mathbf{e}_{123} + 3\mathbf{e}_{013} - 2\mathbf{e}_{021}`$ (the point at
//! $`(0, 3, -2)`$) is the plane
//! $`\mathbf{e}_0 + 3\mathbf{e}_2 - 2\mathbf{e}_3`$.
//!
//! ### Regressive Product (reg)
//!
//! The regressive product is implemented in terms of the exterior product.
//! Given multivectors $`\mathbf{a}`$ and $`\mathbf{b}`$, the regressive product
//! $`\mathbf{a}\vee\mathbf{b}`$ is equivalent to
//! $`J(J(\mathbf{a})\wedge J(\mathbf{b}))`$.
//! Thus, both meets and joins reside in the same algebraic structure.
//!
//! #### example "Joining two points"
//!
//! ```cpp
//!     kln::point p1{x1, y1, z1};
//!     kln::point p2{x2, y2, z2};
//!
//!     // l contains both p1 and p2.
//!     kln::line l = p1 & p2;
//! ```
//!
//! #### example "Joining a line and a point"
//!
//! ```cpp
//!     kln::point p1{x, y, z};
//!     kln::line l2{mx, my, mz, dx, dy, dz};
//!
//!     // p2 contains both p1 and l2.
//!     kln::plane p2 = p1 & l2;
//! ```
//!
//! ### Exponential and Logarithm (exp and log)
//!
//! The group of rotations, translations, and screws (combined rotatation and
//! translation) is _nonlinear_. This means, given say, a rotor $`\mathbf{r}`$,
//! the rotor
//! $`\frac{\mathbf{r}}{2}`$ _does not_ correspond to half the rotation.
//! Similarly, for a motor $`\mathbf{m}`$, the motor $`n \mathbf{m}`$ is not $`n`$
//! applications of the motor $`\mathbf{m}`$. One way we could achieve this is
//! through exponentiation; for example, the motor $`\mathbf{m}^3`$ will perform
//! the screw action of $`\mathbf{m}`$ three times. However, repeated
//! multiplication in this fashion lacks both efficiency and numerical
//! stability.
//!
//! The solution is to take the logarithm of the action which maps the action to
//! a linear space. Using `log(A)` where `A` is one of `rotor`,
//! `translator`, or `motor`, we can apply linear scaling to `log(A)`,
//! and then re-exponentiate the result. Using this technique, `exp(n * log(A))`
//! is equivalent to $`\mathbf{A}^n`$.
//!
//! Takes the principal branch of the logarithm of the motor, returning a
//! bivector. Exponentiation of that bivector without any changes produces
//! this motor again. Scaling that bivector by $`\frac{1}{n}`$,
//! re-exponentiating, and taking the result to the $n$th power will also
//! produce this motor again. The logarithm presumes that the motor is
//! normalized.
//!
//! # Lines
//!
//! Klein provides three line classes: [`Line`], [`Branch`], and [`IdealLine`]. The
//! line class represents a full six-coordinate bivector. The branch contains
//! three non-degenerate components (aka, a line through the origin). The ideal
//! line represents the line at infinity. When the line is created as a meet
//! of two planes or join of two points (or carefully selected Plücker
//! coordinates), it will be a Euclidean line (factorizable as the meet of two vectors).
//!
//! # Motors
//!
//! A `motor` represents a kinematic motion in our algebra. From
//! [Chasles'
//! theorem](https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics)), we
//! know that any rigid body displacement can be produced by a translation along
//! a line, followed or preceded by a rotation about an axis parallel to that
//! line. The motor algebra is isomorphic to the dual quaternions but exists
//! here in the same algebra as all the other geometric entities and actions at
//! our disposal. Operations such as composing a motor with a rotor or
//! translator are possible for example. The primary benefit to using a motor
//! over its corresponding matrix operation is twofold. First, you get the
//! benefit of numerical stability when composing multiple actions via the
//! geometric product (`*`). Second, because the motors constitute a continuous
//! group, they are amenable to smooth interpolation and differentiation.
//!
//! ### example
//!
//! ```c++
//!     // Create a rotor representing a pi/2 rotation about the z-axis
//!     // Normalization is done automatically
//!     rotor r{M_PI * 0.5f, 0.0, 0.0, 1.0};
//!
//!     // Create a translator that represents a translation of 1 unit
//!     // in the yz-direction. Normalization is done automatically.
//!     translator t{1.0, 0.0, 1.0, 1.0};
//!
//!     // Create a motor that combines the action of the rotation and
//!     // translation above.
//!     motor m = r * t;
//!
//!     // Initialize a point at (1, 3, 2)
//!     kln::point p1{1.0, 3.0, 2.0};
//!
//!     // Translate p1 and rotate it to create a new point p2
//!     kln::point p2 = m(p1);
//! ```
//!
//! Motors can be multiplied to one another with the `*` operator to create
//! a new motor equivalent to the application of each factor.
//!
//! ### example
//!
//! ```c++
//!     // Suppose we have 3 motors m1, m2, and m3
//!
//!     // The motor m created here represents the combined action of m1,
//!     // m2, and m3.
//!     kln::motor m = m3 * m2 * m1;
//! ```
//!
//! The same `*` operator can be used to compose the motor's action with other
//! translators and rotors.
//!
//! A demonstration of using the exponential and logarithmic map to blend
//! between two motors is provided in a test case
//! [here](https://github.com/jeremyong/Klein/blob/master/test/test_exp_log.cpp#L48).
//!
//! # Exterior Product (ext/meet)
//!
//! The exterior product between two basis elements extinguishes if the two
//! operands share any common index. Otherwise, the element produced is
//! equivalent to the union of the subspaces. A sign flip is introduced if
//! the concatenation of the element indices is an odd permutation of the
//! cyclic basis representation. The exterior product extends to general
//! multivectors by linearity.
//!
//! # example "Meeting two planes"
//!
//! ```cpp
//!     kln::plane p1{x1, y1, z1, d1};
//!     kln::plane p2{x2, y2, z2, d2};
//!
//!     // l lies at the intersection of p1 and p2.
//!     kln::line l = p1 ^ p2;
//! ```
//!
//! # example "Meeting a line and a plane"
//!
//! ```cpp
//!     kln::plane p1{x, y, z, d};
//!     kln::line l2{mx, my, mz, dx, dy, dz};
//!
//!     // p2 lies at the intersection of p1 and l2.
//!     kln::point p2 = p1 ^ l2;
//! ```
//!
//! Geometric Product (gp)
//!
//! The geometric product extends the exterior product with a notion of a
//! metric. When the subspace intersection of the operands of two basis
//! elements is non-zero, instead of the product extinguishing, the grade
//! collapses and a scalar weight is included in the final result according
//! to the metric. The geometric product can be used to build rotations, and
//! by extension, rotations and translations in projective space.
//!
//! ### example "Rotor composition"
//!
//! ```cpp
//!     kln::rotor r1{ang1, x1, y1, z1};
//!     kln::rotor r2{ang2, x2, y2, z2};
//!
//!     // Compose rotors with the geometric product
//!     kln::rotor r3 = r1 * r2;; // r3 combines r2 and r1 in that order
//! ```
//!
//! ### example "Two reflections"
//!
//! ```cpp
//!     kln::plane p1{x1, y1, z1, d1};
//!     kln::plane p2{x2, y2, z2, d2};
//!
//!     // The geometric product of two planes combines their reflections
//!     kln::motor m3 = p1 * p2; // m3 combines p2 and p1 in that order
//!     // If p1 and p2 were parallel, m3 would be a translation. Otherwise,
//!     // m3 would be a rotation.
//! ```
//!
//! Another common usage of the geometric product is to create a transformation
//! that takes one entity to another. Suppose we have two entities $a$ and $b$
//! and suppose that both entities are normalized such that $`a^2 = b^2 = 1`$.
//! Then, the action created by $`\sqrt{ab}`$ is the action that maps $b$ to $a$.
//!
//! ### example "Motor between two lines"
//!
//! ```cpp
//!     kln::line l1{mx1, my1, mz1, dx1, dy1, dz1};
//!     kln::line l2{mx2, my2, mz2, dx2, dy2, dz2};
//!     // Ensure lines are normalized if they aren't already
//!     l1.normalize();
//!     l2.normalize();
//!     kln::motor m = kln::sqrt(l1 * l2);
//!
//!     kln::line l3 = m(l2);
//!     // l3 will be projectively equivalent to l1.
//! ```
//!
//! Also provided are division operators that multiply the first argument by the
//! inverse of the second argument.
//!
//! Construct a motor $`m`$ such that $`\sqrt{m}`$ takes plane $`b`$ to plane $`a`$.
//!
//! ### example
//!
//! ```cpp
//!     kln::plane p1{x1, y1, z1, d1};
//!     kln::plane p2{x2, y2, z2, d2};
//!     kln::motor m = sqrt(p1 * p2);
//!     plane p3 = m(p2);
//!     // p3 will be approximately equal to p1
//! ```
//!
//! # Symmetric Inner Product (dot)
//!
//! The symmetric inner product takes two arguments and contracts the lower
//! graded element to the greater graded element. If lower graded element
//! spans an index that is not contained in the higher graded element, the
//! result is annihilated. Otherwise, the result is the part of the higher
//! graded element "most unlike" the lower graded element. Thus, the
//! symmetric inner product can be thought of as a bidirectional contraction
//! operator.
//!
//! There is some merit in providing both a left and right contraction
//! operator for explicitness. However, when using Klein, it's generally
//! clear what the interpretation of the symmetric inner product is with
//! respect to the projection on various entities.
//!
//! # Example "Angle between planes"
//!
//! ```cpp
//!     kln::plane a{x1, y1, z1, d1};
//!     kln::plane b{x2, y2, z2, d2};
//!
//!     // Compute the cos of the angle between two planes
//!     float cos_ang = a | b;
//! ```
//!
//! # Example "Line to plane through point"
//!
//! ```cpp
//!     kln::point a{x1, y1, z1};
//!     kln::plane b{x2, y2, z2, d2};
//!
//!     // The line l contains a and the shortest path from a to plane b.
//!     line l = a | b;
//! ```
//!
//! # Translators
//!
//! A translator represents a rigid-body displacement along a normalized axis.
//! To apply the translator to a supported entity, the call operator is
//! available.
//!
//! ## Example
//!
//! ```c++
//!     // Initialize a point at (1, 3, 2)
//!     kln::point p{1.0, 3.0, 2.0};
//!
//!     // Create a normalized translator representing a 4-unit
//!     // displacement along the xz-axis.
//!     kln::translator r{4.0, 1.0, 0.0, 1.0};
//!
//!     // Displace our point using the created translator
//!     kln::point translated = r(p);
//! ```
//! We can translate lines and planes as well using the translator's call
//! operator.
//!
//! Translators can be multiplied to one another with the `*` operator to create
//! a new translator equivalent to the application of each factor.
//!
//! ## Example
//!
//! ```c++
//!     // Suppose we have 3 translators t1, t2, and t3
//!
//!     // The translator t created here represents the combined action of
//!     // t1, t2, and t3.
//!     kln::translator t = t3 * t2 * t1;
//! ```
//!
//! The same `*` operator can be used to compose the translator's action with
//! other rotors and motors.
//!
//! # Rotors
//!
//! The rotor is an entity that represents a rigid rotation about an axis.
//! To apply the rotor to a supported entity, the call operator is available.
//!
//! ### example
//!
//! ```c++
//!     // Initialize a point at (1, 3, 2)
//!     kln::point p{1.f, 3.f, 2.f};
//!
//!     // Create a normalized rotor representing a pi/2 radian
//!     // rotation about the xz-axis.
//!     kln::rotor r{M_PI * 0.5f, 1.f, 0.f, 1.f};
//!
//!     // Rotate our point using the created rotor
//!     kln::point rotated = r(p);
//! ```
//! We can rotate lines and planes as well using the rotor's call operator.
//!
//! Rotors can be multiplied to one another with the `*` operator to create
//! a new rotor equivalent to the application of each factor.
//!
//! ### example
//!
//! ```c++
//!     // Create a normalized rotor representing a $\frac{\pi}{2}$ radian
//!     // rotation about the xz-axis.
//!     kln::rotor r1{M_PI * 0.5f, 1.f, 0.f, 1.f};
//!
//!     // Create a second rotor representing a $\frac{\pi}{3}$ radian
//!     // rotation about the yz-axis.
//!     kln::rotor r2{M_PI / 3.f, 0.f, 1.f, 1.f};
//!
//!     // Use the geometric product to create a rotor equivalent to first
//!     // applying r1, then applying r2. Note that the order of the
//!     // operands here is significant.
//!     kln::rotor r3 = r2 * r1;
//! ```
//!
//! The same `*` operator can be used to compose the rotor's action with other
//! translators and motors.

#![feature(stdarch)]
#![allow(deprecated, non_snake_case, unused_unsafe)]
#![warn(clippy::all)]

#[macro_use]
pub mod arch;


mod join; // f32x4
mod exp_log; // f32x4
mod multivector_ep;
mod multivector_gp; // f32x4
mod multivector_ip; // f32x4

mod direction; // done f32x4
mod dual; // done scalar
mod line; // done
mod matrix;
mod motor;
mod plane; // done
mod point; // done
mod rotor;
mod translator; // done

mod macros;

pub use self::{
    direction::Direction,
    dual::Dual,
    line::{Branch, IdealLine, Line},
    matrix::{Mat3x4, Mat4x4},
    motor::Motor,
    plane::Plane,
    point::{Origin, Point},
    rotor::Rotor,
    translator::Translator,
};

/*
pub fn direction(x: f32, y: f32, z: f32) -> Direction {
    Direction::new(x, y, z)
}

pub fn dual(p: f32, q: f32) -> Dual {
    Dual::new(p, q)
}

pub fn branch(a: f32, b: f32, c: f32) -> Branch {
    Branch::new(a, b, c)
}

pub fn ideal_line(a: f32, b: f32, c: f32) -> IdealLine {
    IdealLine::new(a, b, c)
}

pub fn line(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Line {
    Line::new(a, b, c, d, e, f)
}

pub fn motor(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Motor {
    Motor::new(a, b, c, d, e, f, g, h)
}

pub fn plane(a: f32, b: f32, c: f32, d: f32) -> Plane {
    Plane::new(a, b, c, d)
}

pub fn point(x: f32, y: f32, z: f32) -> Point {
    Point::new(x, y, z)
}

pub fn rotor(a: f32, b: f32, c: f32, d: f32) -> Rotor {
    Rotor::new(a, b, c, d)
}

pub fn translator(delta: f32, x: f32, y: f32, z: f32) -> Translator {
    Translator::new(delta, x, y, z)
}
*/
