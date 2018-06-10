#![deny(missing_docs)]

//! A crate for things that are
//! 1) Lazily initialized
//! 2) Expensive to create
//! 3) Immutable after creation
//! 4) Used on multiple threads
//!
//! `Lazy<T>` is better than `Mutex<Option<T>>` because after creation accessing
//! `T` does not require any locking, just a single boolean load with
//! `Ordering::Acquire` (which on x86 is just a compiler barrier, not an actual
//! memory barrier).

use std::cell::UnsafeCell;
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

enum ThisOrThat<T, U> {
    This(T),
    That(U),
}

/// `LazyTransform<T, U>` is a synchronized holder type, that holds a value of
/// type T until it is lazily converted into a value of type U.
pub struct LazyTransform<T, U> {
    initialized: AtomicBool,
    lock: Mutex<()>,
    value: UnsafeCell<Option<ThisOrThat<T, U>>>,
}

// Implementation details.
impl<T, U> LazyTransform<T, U>
    where T: Sync,
          U: Sync
{
    fn extract(&self) -> Option<&U> {
        // Make sure we're initialized first!
        match unsafe { (*self.value.get()).as_ref() } {
            None => None,
            Some(&ThisOrThat::This(_)) => panic!(), // Should already be initialized!
            Some(&ThisOrThat::That(ref that)) => Some(that),
        }
    }
}

// Public API.
impl<T, U> LazyTransform<T, U> {
    /// Construct a new, untransformed `LazyTransform<T, U>` with an argument of
    /// type T.
    pub fn new(t: T) -> LazyTransform<T, U> {
        LazyTransform {
            initialized: AtomicBool::new(false),
            lock: Mutex::new(()),
            value: UnsafeCell::new(Some(ThisOrThat::This(t))),
        }
    }

    /// Unwrap the contained value, returning `Ok(U)` if the `LazyTransform<T, U>` has been
    /// transformed or `Err(T)` if it has not.
    pub fn into_inner(self) -> Result<U, T> {
        // We don't need to inspect `self.initialized` since `self` is owned
        // so it is guaranteed that no other threads are accessing its data.
        match self.value.into_inner().unwrap() {
            ThisOrThat::This(t) => Err(t),
            ThisOrThat::That(u) => Ok(u),
        }
    }
}

// Public API.
impl<T, U> LazyTransform<T, U>
    where T: Sync,
          U: Sync
{
    /// Get a reference to the transformed value, invoking `f` to transform it
    /// if the `LazyTransform<T, U>` has yet to be transformed.  It is
    /// guaranteed that if multiple calls to `get_or_create` race, only one
    /// will invoke its closure, and every call will receive a reference to the
    /// newly transformed value.
    ///
    /// The closure can only ever be called once, so think carefully about what
    /// transformation you want to apply!
    pub fn get_or_create<F>(&self, f: F) -> &U
        where F: FnOnce(T) -> U
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
                let this = match value.take().unwrap() {
                    ThisOrThat::This(t) => t,
                    ThisOrThat::That(_) => panic!(), // Can't already be initialized!
                };
                *value = Some(ThisOrThat::That(f(this)));
                self.initialized.store(true, Ordering::Release);
            } else {
                // We raced, and someone else initialized us. We can fall
                // through now.
            }
        }

        // We're initialized, our value is immutable, no synchronization needed.
        self.extract().unwrap()
    }

    /// Get a reference to the transformed value, returning `Some(&U)` if the
    /// `LazyTransform<T, U>` has been transformed or `None` if it has not.  It
    /// is guaranteed that if a reference is returned it is to the transformed
    /// value inside the the `LazyTransform<T>`.
    pub fn get(&self) -> Option<&U> {
        if self.initialized.load(Ordering::Acquire) {
            // We're initialized, our value is immutable, no synchronization needed.
            self.extract()
        } else {
            None
        }
    }
}

unsafe impl<T, U> Sync for LazyTransform<T, U>
    where T: Sync + Send,
          U: Sync
{
}

impl<T, U> Default for LazyTransform<T, U>
    where T: Sync + Default,
          U: Sync
{
    fn default() -> Self {
        LazyTransform::new(T::default())
    }
}

/// `Lazy<T>` is a lazily initialized synchronized holder type.  You can think
/// of it as a `LazyTransform` where the initial type doesn't exist.
pub struct Lazy<T> {
    inner: LazyTransform<(), T>,
}

impl<T> Lazy<T> {
    /// Construct a new, uninitialized `Lazy<T>`.
    pub fn new() -> Lazy<T> {
        Self::default()
    }

    /// Unwrap the contained value, returning `Some` if the `Lazy<T>` has been initialized
    /// or `None` if it has not.
    pub fn into_inner(self) -> Option<T> {
        self.inner.into_inner().ok()
    }
}

impl<T> Lazy<T>
    where T: Sync
{
    /// Get a reference to the contained value, invoking `f` to create it
    /// if the `Lazy<T>` is uninitialized.  It is guaranteed that if multiple
    /// calls to `get_or_create` race, only one will invoke its closure, and
    /// every call will receive a reference to the newly created value.
    ///
    /// The value stored in the `Lazy<T>` is immutable after the closure returns
    /// it, so think carefully about what you want to put inside!
    pub fn get_or_create<F>(&self, f: F) -> &T
        where F: FnOnce() -> T
    {
        self.inner.get_or_create(|_| f())
    }

    /// Get a reference to the contained value, returning `Some(ref)` if the
    /// `Lazy<T>` has been initialized or `None` if it has not.  It is
    /// guaranteed that if a reference is returned it is to the value inside
    /// the `Lazy<T>`.
    pub fn get(&self) -> Option<&T> {
        self.inner.get()
    }
}

// `#[derive(Default)]` automatically adds `T: Default` trait bound, but that
// is too restrictive, because `Lazy<T>` always has a default value for any `T`.
impl<T> Default for Lazy<T> {
    fn default() -> Self {
        Lazy { inner: LazyTransform::new(()) }
    }
}

impl<T> fmt::Debug for Lazy<T>
    where T: fmt::Debug + Sync
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(v) = self.get() {
            f.write_fmt(format_args!("Lazy({:?})", v))
        } else {
            f.write_str("Lazy(<uninitialized>)")
        }
    }
}

#[cfg(test)]
extern crate scoped_pool;

#[cfg(test)]
mod tests {

    use scoped_pool::Pool;
    use std::{thread, time};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use super::{Lazy, LazyTransform};

    #[test]
    fn test_lazy() {
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

    #[test]
    fn test_lazy_transform() {
        let lazy_value: LazyTransform<u8, u8> = LazyTransform::new(21);

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

                    let value = *lazy_ref.get_or_create(|v| {
                        // Make everybody else wait on me, because I'm a jerk.
                        thread::sleep(ten_millis);

                        // Make this relaxed so it doesn't interfere with
                        // Lazy internals at all.
                        n_ref.fetch_add(1, Ordering::Relaxed);

                        v * 2
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
