fn double_deref(pp: *mut *mut i32) -> i32 {
    return (*(*pp));
}
