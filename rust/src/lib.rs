// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[macro_export]
macro_rules! cuda_error {
    () => {
        // Declare C/C++ counterpart as following:
        // extern "C" { fn foobar(...) -> cuda::Error; }
        mod cuda {
            #[repr(C)]
            pub struct Error {
                pub code: i32,
                str: Option<core::ptr::NonNull<i8>>, // just strdup("string") from C/C++
            }

            impl Drop for Error {
                fn drop(&mut self) {
                    extern "C" {
                        fn free(str: Option<core::ptr::NonNull<i8>>);
                    }
                    unsafe { free(self.str) };
                    self.str = None;
                }
            }

            impl From<Error> for String {
                fn from(status: Error) -> Self {
                    let c_str = if let Some(ptr) = status.str {
                        unsafe { std::ffi::CStr::from_ptr(ptr.as_ptr()) }
                    } else {
                        extern "C" {
                            fn cudaGetErrorString(code: i32) -> *const i8;
                        }
                        unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(status.code)) }
                    };
                    String::from(c_str.to_str().unwrap_or("unintelligible"))
                }
            }
        }
    };
}

use core::ffi::c_void;
use core::mem::transmute;

#[repr(C)]
pub struct Gpu_Ptr<T> {
    ptr: *const c_void,
    phantom: core::marker::PhantomData<T>,
}

#[cfg(feature = "cuda")]
impl<T> Drop for Gpu_Ptr<T> {
    fn drop(&mut self) {
        extern "C" {
            fn drop_gpu_ptr_t(by_ref: &Gpu_Ptr<c_void>);
        }
        unsafe { drop_gpu_ptr_t(transmute::<&Gpu_Ptr<T>, &Gpu_Ptr<c_void>>(self)) };
        self.ptr = core::ptr::null();
    }
}

#[cfg(feature = "cuda")]
impl<T> Clone for Gpu_Ptr<T> {
    fn clone(&self) -> Self {
        extern "C" {
            fn clone_gpu_ptr_t(ret: &mut Gpu_Ptr<c_void>, by_ref: &Gpu_Ptr<c_void>);
        }
        let mut ret = Self {
            ptr: core::ptr::null(),
            phantom: core::marker::PhantomData,
        };
        unsafe {
            clone_gpu_ptr_t(
                transmute::<&mut Gpu_Ptr<T>, &mut Gpu_Ptr<c_void>>(&mut ret),
                transmute::<&Gpu_Ptr<T>, &Gpu_Ptr<c_void>>(self),
            )
        };
        ret
    }
}
