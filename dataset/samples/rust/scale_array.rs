fn scale_array(arr: *mut i32, n: i32, factor: i32) {
    let mut i: i32 = 0;
while (i < n) { arr[i as usize] *= factor;
    { let _v = i; i += 1; _v };
}
}
