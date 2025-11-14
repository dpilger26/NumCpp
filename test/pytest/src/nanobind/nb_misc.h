/*
    nanobind/nb_misc.h: Miscellaneous bits (GIL, etc.)

    Copyright (c) 2022 Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

NAMESPACE_BEGIN(NB_NAMESPACE)

struct gil_scoped_acquire {
public:
    NB_NONCOPYABLE(gil_scoped_acquire)
    gil_scoped_acquire() noexcept : state(PyGILState_Ensure()) { }
    ~gil_scoped_acquire() { PyGILState_Release(state); }

private:
    const PyGILState_STATE state;
};

struct gil_scoped_release {
public:
    NB_NONCOPYABLE(gil_scoped_release)
    gil_scoped_release() noexcept : state(PyEval_SaveThread()) { }
    ~gil_scoped_release() { PyEval_RestoreThread(state); }

private:
    PyThreadState *state;
};

struct ft_mutex {
public:
    NB_NONCOPYABLE(ft_mutex)
    ft_mutex() = default;

#if !defined(NB_FREE_THREADED)
    void lock() { }
    void unlock() { }
#else
    void lock() { PyMutex_Lock(&mutex); }
    void unlock() { PyMutex_Unlock(&mutex); }
private:
    PyMutex mutex { 0 };
#endif
};

struct ft_lock_guard {
public:
    NB_NONCOPYABLE(ft_lock_guard)
    ft_lock_guard(ft_mutex &m) : m(m) { m.lock(); }
    ~ft_lock_guard() { m.unlock(); }
private:
    ft_mutex &m;
};


struct ft_object_guard {
public:
    NB_NONCOPYABLE(ft_object_guard)
#if !defined(NB_FREE_THREADED)
    ft_object_guard(handle) { }
#else
    ft_object_guard(handle h) { PyCriticalSection_Begin(&cs, h.ptr()); }
    ~ft_object_guard() { PyCriticalSection_End(&cs); }
private:
    PyCriticalSection cs;
#endif
};

struct ft_object2_guard {
public:
    NB_NONCOPYABLE(ft_object2_guard)
#if !defined(NB_FREE_THREADED)
    ft_object2_guard(handle, handle) { }
#else
    ft_object2_guard(handle h1, handle h2) { PyCriticalSection2_Begin(&cs, h1.ptr(), h2.ptr()); }
    ~ft_object2_guard() { PyCriticalSection2_End(&cs); }
private:
    PyCriticalSection2 cs;
#endif
};

inline bool leak_warnings() noexcept {
    return detail::leak_warnings();
}

inline bool implicit_cast_warnings() noexcept {
    return detail::implicit_cast_warnings();
}

inline void set_leak_warnings(bool value) noexcept {
    detail::set_leak_warnings(value);
}

inline void set_implicit_cast_warnings(bool value) noexcept {
    detail::set_implicit_cast_warnings(value);
}

inline dict globals() {
    PyObject *p = PyEval_GetGlobals();
    if (!p)
        raise("nanobind::globals(): no frame is currently executing!");
    return borrow<dict>(p);
}

inline Py_hash_t hash(handle h) {
    Py_hash_t rv = PyObject_Hash(h.ptr());
    if (rv == -1 && PyErr_Occurred())
        nanobind::raise_python_error();
    return rv;
}

inline bool isinstance(handle inst, handle cls) {
    int ret = PyObject_IsInstance(inst.ptr(), cls.ptr());
    if (ret == -1)
      nanobind::raise_python_error();
    return ret;
}

inline bool is_alive() noexcept {
    return detail::is_alive();
}

NAMESPACE_END(NB_NAMESPACE)
