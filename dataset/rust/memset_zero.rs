fn memset_zero(buf: *mut i8, n: i32) {
    let mut i: i32 = 0;
while (i < n) { buf[i as usize] = 0;
    { let _v = i; i += 1; _v };
}
}
