fn ptr_offset(arr: *mut i32, idx: i32) -> i32 {
    return (*(arr + idx));
}
