use klein::{Motor, Point};

struct Joint {
    inv_bind_pose: Motor,
    parent_offset: u8,
    group_size: u8,
}

struct Skeleton {
    joints: Vec<(Joint, String)>,
}

struct Pose {
    // Array of poses for each joint
    joint_poses: Vec<Motor>,
    // NOTE: some engines allow animators to include scale data with each joint
    // pose, but we'll ignore that for the time being.
}

struct Clip {
    poses: Vec<Pose>, // Array of poses
    //uint16_t size;         // Number of poses in the animation clip
    timestamps: u16,   // Array of timestamps for each skeletal pose
    timestamp_us: u32, // Conversion from timestamp to microseconds
}

/*
struct SkeletonInstance {
    // All positions here are in world coordinate space
    joint_positions: Vec<Point>,
    world_location: Point,
}

// NOTE: t is expected to be between 0 and 1
fn nlerp(m1: Motor, m2: Motor, t: f32) -> Motor {
    ((1.0 - t) * m1 + t * m2).normalize()
}
// Blend between two motors with a parameter t in the range [0, 1]
fn slerp(a: Motor, b: Motor, t: f32) -> Motor {
    // Starting from a, the motor needed to get to b is b * ~a.
    // To perform this motion continuously, we can take the principal
    // branch of the logarithm of b * ~a, and subdivide it before
    // re-exponentiating it to produce a motor again.

    // In practice, this should be cached whenever possible.
    // exp(log(m)) = exp(t*log(m) + (1 - t)*log(m))
    // = exp(t*(log(m))) * exp((1 - t)*log(m))
    let motor_step: Line = (b * a.reverse()).log() * t;

    // The exponential of the step here can be cached if the blend occurs
    // with fixed steps toward the final motor. Compose the interpolated
    // result with the start motor to produce the intermediate blended motor.
    motor_step.exp() * a
}

fn animate_keyframe(parent: &Sceleton, instance: &mut SkeletonInstance, target: &Pose) {
    // We need to write out the final transforms to the instance of the parent
    // skeleton. The clip is the set of joint poses we need to apply.

    // First, initialize the position of every joint to the world location
    for position in &mut instance.joint_positions {
        *position = instance.world_location;
    }

    // Locomoting the world location of the instance according to the animation
    // clip is known as "root motion" animation and is a relatively common
    // technique, although it does have some tradeoffs outside the scope of this
    // tutorial.

    // For each joint, apply its corresponding joint pose motor to every
    // position in its group.
    for (uint16_t i = 0; i != parent.size; ++i)
    {
        // To apply the joint pose motor, we use the call operator. Here, we
        // use the overload that is efficient when applying the same motor to
        // a set of different positions.
        target.joint_poses[i](
            &instances.joint_positions[i], // Position input
            &instances.joint_positions[i], // Position output
            parent.joints[i].group_size);  // Count
    }
}

// Given a skeleton, an instance of the skeleton, a clip, and a timestamp,
// transform the instance to the correct pose sampled from the clip.

fn animate_sample(
    parent: &Sceleton,
    instance: &mut SkeletonInstance,
    active_clip: &Clip,
    timestamp_ms: i32,
    // scratch is a mutable pose with sufficient memory
    // to hold our interpolated joint poses.
    scratch: &Pose,
) {
    pose*  previous;
    pose*  next;
    float* t;
    // This function isn't provided, but it takes a clip and timestamp
    // and produces the poses that straddle the requested time and the
    // interpolation parameter.
    query_pose_endpoints(clip, timestamp, &previous, &next, &t);

    for (uint16_t i = 0; i != parent.size; ++i)
    {
        // This could use slerp or nlerp if we wanted. A possible
        // implementation of this slerp function was given above.
        scratch.joint_poses[i] = slerp(
            previous->joint_poses[i],
            next->joint_poses[i],
            *t
        );
    }

    // Reuse our keyframe forward kinematic routine from above
    animate_keyframe(parent, instance, scratch);
}
*/

fn main() {

}