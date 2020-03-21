#[test]
fn rcp_nr1() {
    use klein::arch::f32x4;

    let buf = f32x4::new(4.0, 3.0, 2.0, 1.0).rcp_nr1().into_array();
    approx::abs_diff_eq!(buf[0], 1.0);
    approx::abs_diff_eq!(buf[1], 0.5);
    approx::abs_diff_eq!(buf[2], 1.0 / 3.0);
    approx::abs_diff_eq!(buf[3], 0.25);
}
