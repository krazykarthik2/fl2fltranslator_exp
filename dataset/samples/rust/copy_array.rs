fn copy_array(dst: *mut i32, src: *mut i32, n: i32) {
    let mut i: i32 = 0;
while (i < n) { dst[i as usize] = src[i as usize];
    { let _v = i; i += 1; _v };
}
}
