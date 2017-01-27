#![deny(missing_docs)]

//! A crate for things that are
//! 1) Lazily initialized
//! 2) Expensive to create
//! 3) Immutable after creation
//! 4) Used on multiple threads
//!
//! Lazy<T> is better than Mutex<Option<T>> because after creation accessing
//! T does not require any locking, just a single boolean load with
//! Ordering::Acquire (which on x86 is just a compiler barrier, not an actual
//! memory barrier).

use std::cell::UnsafeCell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// `Lazy<T>` is a lazily initialized synchronized holder type.
pub struct Lazy<T> {
    initialized: AtomicBool,
    lock: Mutex<()>,
    value: UnsafeCell<Option<T>>,
}

impl<T> Lazy<T> {
    /// Construct a new, uninitialized `Lazy<T>`.
    pub fn new() -> Lazy<T> {
        Lazy {
            initialized: AtomicBool::new(false),
            lock: Mutex::new(()),
            value: UnsafeCell::new(None),
        }
    }

    /// Get a reference to the contained value, invoking `f` to create it
    /// if the `Lazy<T>` is uninitialized.  It is guaranteed that if multiple
    /// calls to `get_or_create` race, only one will invoke its closure, and
    /// every call will receive a reference to the newly created value.
    ///
    /// The value stored in the `Lazy<T>` is immutable after the closure returns
    /// it, so think carefully about what you want to put inside!
    pub fn get_or_create<'a, F>(&'a self, f: F) -> &'a T
        where F: FnOnce() -> T
    {
        // In addition to being correct, this pattern is vouched for by Hans Boehm
        // (http://schd.ws/hosted_files/cppcon2016/74/HansWeakAtomics.pdf Page 27)
        if !self.initialized.load(Ordering::Acquire) {
            // We *may* not be initialized. We have to block to be certain.
            let _lock = self.lock.lock().unwrap();
            if !self.initialized.load(Ordering::Relaxed) {
                // Ok, we're definitely uninitialized.
                // Safe to fiddle with the UnsafeCell now, because we're locked,
                // and there can't be any outstanding references.
                let value = unsafe { &mut *self.value.get() };
                *value = Some(f());
                self.initialized.store(true, Ordering::Release);
            } else {
                // We raced, and someone else initialized us. We can fall
                // through now.
            }
        }

        // We're initialized, our value is immutable, no synchronization needed.
        unsafe { (*self.value.get()).as_ref().unwrap() }
    }

    /// Get a reference to the contained value, returning `Some(ref)` if the
    /// `Lazy<T>` has been initialized or `None` if it has not.  It is
    /// guaranteed that if a reference is returned it is to the value inside
    /// the `Lazy<T>`.
    pub fn get<'a>(&'a self) -> Option<&'a T> {
        if self.initialized.load(Ordering::Acquire) {
            // We're initialized, our value is immutable, no synchronization needed.
            unsafe { (*self.value.get()).as_ref() }
        } else {
            None
        }
    }
}

unsafe impl<T> Sync for Lazy<T> { }

#[cfg(test)]
extern crate scoped_pool;

#[cfg(test)]
mod tests {

    use scoped_pool::Pool;
    use std::{thread, time};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use super::Lazy;

    #[test]
    fn test_basic() {
        let lazy_value: Lazy<u8> = Lazy::new();

        assert_eq!(lazy_value.get(), None);

        let n = AtomicUsize::new(0);

        let pool = Pool::new(100);
        pool.scoped(|scope| {
            for _ in 0..100 {
                let lazy_ref = &lazy_value;
                let n_ref = &n;
                scope.execute(move || {
                    let ten_millis = time::Duration::from_millis(10);
                    thread::sleep(ten_millis);

                    let value = *lazy_ref.get_or_create(|| {
                        // Make everybody else wait on me, because I'm a jerk.
                        thread::sleep(ten_millis);

                        // Make this relaxed so it doesn't interfere with
                        // Lazy internals at all.
                        n_ref.fetch_add(1, Ordering::Relaxed);

                        42
                    });
                    assert_eq!(value, 42);

                    let value = lazy_ref.get();
                    assert_eq!(value, Some(&42));
                });
            }
        });

        assert_eq!(n.load(Ordering::SeqCst), 1);
    }
}
