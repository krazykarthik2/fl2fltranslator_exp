fn swap(a: *mut i32, b: *mut i32) {
    let mut t: i32 = (*a);
    (*a) = (*b);
    (*b) = t;
}
