/*
    nanobind/nb_attr.h: Annotations for function and class declarations

    Copyright (c) 2022 Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

NAMESPACE_BEGIN(NB_NAMESPACE)

struct scope {
    PyObject *value;
    NB_INLINE scope(handle value) : value(value.ptr()) {}
};

struct name {
    const char *value;
    NB_INLINE name(const char *value) : value(value) {}
};

struct arg_v;
struct arg_locked;
struct arg_locked_v;

// Basic function argument descriptor (no default value, not locked)
struct arg {
    NB_INLINE constexpr explicit arg(const char *name = nullptr) : name_(name), signature_(nullptr) { }

    // operator= can be used to provide a default value
    template <typename T> NB_INLINE arg_v operator=(T &&value) const;

    // Mutators that don't change default value or locked state
    NB_INLINE arg &noconvert(bool value = true) {
        convert_ = !value;
        return *this;
    }
    NB_INLINE arg &none(bool value = true) {
        none_ = value;
        return *this;
    }
    NB_INLINE arg &sig(const char *value) {
        signature_ = value;
        return *this;
    }

    // After lock(), this argument is locked
    NB_INLINE arg_locked lock();

    const char *name_, *signature_;
    uint8_t convert_{ true };
    bool none_{ false };
};

// Function argument descriptor with default value (not locked)
struct arg_v : arg {
    object value;
    NB_INLINE arg_v(const arg &base, object &&value)
        : arg(base), value(std::move(value)) {}

  private:
    // Inherited mutators would slice off the default, and are not generally needed
    using arg::noconvert;
    using arg::none;
    using arg::sig;
    using arg::lock;
};

// Function argument descriptor that is locked (no default value)
struct arg_locked : arg {
    NB_INLINE constexpr explicit arg_locked(const char *name = nullptr) : arg(name) { }
    NB_INLINE constexpr explicit arg_locked(const arg &base) : arg(base) { }

    // operator= can be used to provide a default value
    template <typename T> NB_INLINE arg_locked_v operator=(T &&value) const;

    // Mutators must be respecified in order to not slice off the locked status
    NB_INLINE arg_locked &noconvert(bool value = true) {
        convert_ = !value;
        return *this;
    }
    NB_INLINE arg_locked &none(bool value = true) {
        none_ = value;
        return *this;
    }
    NB_INLINE arg_locked &sig(const char *value) {
        signature_ = value;
        return *this;
    }

    // Redundant extra lock() is allowed
    NB_INLINE arg_locked &lock() { return *this; }
};

// Function argument descriptor that is potentially locked and has a default value
struct arg_locked_v : arg_locked {
    object value;
    NB_INLINE arg_locked_v(const arg_locked &base, object &&value)
        : arg_locked(base), value(std::move(value)) {}

  private:
    // Inherited mutators would slice off the default, and are not generally needed
    using arg_locked::noconvert;
    using arg_locked::none;
    using arg_locked::sig;
    using arg_locked::lock;
};

NB_INLINE arg_locked arg::lock() { return arg_locked{*this}; }

template <typename... Ts> struct call_guard {
    using type = detail::tuple<Ts...>;
};

struct dynamic_attr {};
struct is_weak_referenceable {};
struct is_method {};
struct is_implicit {};
struct is_operator {};
struct is_arithmetic {};
struct is_flag {};
struct is_final {};
struct is_generic {};
struct kw_only {};
struct lock_self {};

template <size_t /* Nurse */, size_t /* Patient */> struct keep_alive {};
template <typename T> struct supplement {};
template <typename T> struct intrusive_ptr {
    intrusive_ptr(void (*set_self_py)(T *, PyObject *) noexcept)
        : set_self_py(set_self_py) { }
    void (*set_self_py)(T *, PyObject *) noexcept;
};

struct type_slots {
    type_slots (const PyType_Slot *value) : value(value) { }
    const PyType_Slot *value;
};

struct type_slots_callback {
    using cb_t = void (*)(const detail::type_init_data *t,
                          PyType_Slot *&slots, size_t max_slots) noexcept;
    type_slots_callback(cb_t callback) : callback(callback) { }
    cb_t callback;
};

struct sig {
    const char *value;
    sig(const char *value) : value(value) { }
};

struct is_getter { };

template <typename Policy> struct call_policy final {};

NAMESPACE_BEGIN(literals)
constexpr arg operator""_a(const char *name, size_t) { return arg(name); }
NAMESPACE_END(literals)

NAMESPACE_BEGIN(detail)

enum class func_flags : uint32_t {
    /* Low 3 bits reserved for return value policy */

    /// Did the user specify a name for this function, or is it anonymous?
    has_name = (1 << 4),
    /// Did the user specify a scope in which this function should be installed?
    has_scope = (1 << 5),
    /// Did the user specify a docstring?
    has_doc = (1 << 6),
    /// Did the user specify nb::arg/arg_v annotations for all arguments?
    has_args = (1 << 7),
    /// Does the function signature contain an *args-style argument?
    has_var_args = (1 << 8),
    /// Does the function signature contain an *kwargs-style argument?
    has_var_kwargs = (1 << 9),
    /// Is this function a method of a class?
    is_method = (1 << 10),
    /// Is this function a method called __init__? (automatically generated)
    is_constructor = (1 << 11),
    /// Can this constructor be used to perform an implicit conversion?
    is_implicit = (1 << 12),
    /// Is this function an arithmetic operator?
    is_operator = (1 << 13),
    /// When the function is GCed, do we need to call func_data_prelim::free_capture?
    has_free = (1 << 14),
    /// Should the func_new() call return a new reference?
    return_ref = (1 << 15),
    /// Does this overload specify a custom function signature (for docstrings, typing)
    has_signature = (1 << 16),
    /// Does this function potentially modify the elements of the PyObject*[] array
    /// representing its arguments? (nb::keep_alive() or call_policy annotations)
    can_mutate_args = (1 << 17)
};

enum cast_flags : uint8_t {
    // Enable implicit conversions (code assumes this has value 1, don't reorder..)
    convert = (1 << 0),

    // Passed to the 'self' argument in a constructor call (__init__)
    construct = (1 << 1),

    // Indicates that the function dispatcher should accept 'None' arguments
    accepts_none = (1 << 2),

    // Indicates that this cast is performed by nb::cast or nb::try_cast.
    // This implies that objects added to the cleanup list may be
    // released immediately after the caster's final output value is
    // obtained, i.e., before it is used.
    manual = (1 << 3)
};


struct arg_data {
    const char *name;
    const char *signature;
    PyObject *name_py;
    PyObject *value;
    uint8_t flag;
};

struct func_data_prelim_base {
    // A small amount of space to capture data used by the function/closure
    void *capture[3];

    // Callback to clean up the 'capture' field
    void (*free_capture)(void *);

    /// Implementation of the function call
    PyObject *(*impl)(void *, PyObject **, uint8_t *, rv_policy,
                      cleanup_list *);

    /// Function signature description
    const char *descr;

    /// C++ types referenced by 'descr'
    const std::type_info **descr_types;

    /// Supplementary flags
    uint32_t flags;

    /// Total number of parameters accepted by the C++ function; nb::args
    /// and nb::kwargs parameters are counted as one each. If the
    /// 'has_args' flag is set, then there is one arg_data structure
    /// for each of these.
    uint16_t nargs;

    /// Number of parameters to the C++ function that may be filled from
    /// Python positional arguments without additional ceremony.
    /// nb::args and nb::kwargs parameters are not counted in this total, nor
    /// are any parameters after nb::args or after a nb::kw_only annotation.
    /// The parameters counted here may be either named (nb::arg("name")) or
    /// unnamed (nb::arg()).  If unnamed, they are effectively positional-only.
    /// nargs_pos is always <= nargs.
    uint16_t nargs_pos;

    // ------- Extra fields -------

    const char *name;
    const char *doc;
    PyObject *scope;
};

template<size_t Size> struct func_data_prelim : func_data_prelim_base {
    arg_data args[Size];
};

template<> struct func_data_prelim<0> : func_data_prelim_base {};


template <typename F>
NB_INLINE void func_extra_apply(F &f, const name &name, size_t &) {
    f.name = name.value;
    f.flags |= (uint32_t) func_flags::has_name;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, const scope &scope, size_t &) {
    f.scope = scope.value;
    f.flags |= (uint32_t) func_flags::has_scope;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, const sig &s, size_t &) {
    f.flags |= (uint32_t) func_flags::has_signature;
    f.name = s.value;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, const char *doc, size_t &) {
    f.doc = doc;
    f.flags |= (uint32_t) func_flags::has_doc;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, is_method, size_t &) {
    f.flags |= (uint32_t) func_flags::is_method;
}

template <typename F>
NB_INLINE void func_extra_apply(F &, is_getter, size_t &) { }

template <typename F>
NB_INLINE void func_extra_apply(F &f, is_implicit, size_t &) {
    f.flags |= (uint32_t) func_flags::is_implicit;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, is_operator, size_t &) {
    f.flags |= (uint32_t) func_flags::is_operator;
}

template <typename F>
NB_INLINE void func_extra_apply(F &f, rv_policy pol, size_t &) {
    f.flags = (f.flags & ~0b111) | (uint16_t) pol;
}

template <typename F>
NB_INLINE void func_extra_apply(F &, std::nullptr_t, size_t &) { }

template <typename F>
NB_INLINE void func_extra_apply(F &f, const arg &a, size_t &index) {
    uint8_t flag = 0;
    if (a.none_)
        flag |= (uint8_t) cast_flags::accepts_none;
    if (a.convert_)
        flag |= (uint8_t) cast_flags::convert;

    arg_data &arg = f.args[index];
    arg.flag = flag;
    arg.name = a.name_;
    arg.signature = a.signature_;
    arg.value = nullptr;
    index++;
}
// arg_locked will select the arg overload; the locking is added statically
// in nb_func.h

template <typename F>
NB_INLINE void func_extra_apply(F &f, const arg_v &a, size_t &index) {
    arg_data &ad = f.args[index];
    func_extra_apply(f, (const arg &) a, index);
    ad.value = a.value.ptr();
}
template <typename F>
NB_INLINE void func_extra_apply(F &f, const arg_locked_v &a, size_t &index) {
    arg_data &ad = f.args[index];
    func_extra_apply(f, (const arg_locked &) a, index);
    ad.value = a.value.ptr();
}

template <typename F>
NB_INLINE void func_extra_apply(F &, kw_only, size_t &) {}

template <typename F>
NB_INLINE void func_extra_apply(F &, lock_self, size_t &) {}

template <typename F, typename... Ts>
NB_INLINE void func_extra_apply(F &, call_guard<Ts...>, size_t &) {}

template <typename F, size_t Nurse, size_t Patient>
NB_INLINE void func_extra_apply(F &f, nanobind::keep_alive<Nurse, Patient>, size_t &) {
    f.flags |= (uint32_t) func_flags::can_mutate_args;
}

template <typename F, typename Policy>
NB_INLINE void func_extra_apply(F &f, call_policy<Policy>, size_t &) {
    f.flags |= (uint32_t) func_flags::can_mutate_args;
}

template <typename... Ts> struct func_extra_info {
    using call_guard = void;
    static constexpr bool pre_post_hooks = false;
    static constexpr size_t nargs_locked = 0;
};

template <typename T, typename... Ts> struct func_extra_info<T, Ts...>
    : func_extra_info<Ts...> { };

template <typename... Cs, typename... Ts>
struct func_extra_info<call_guard<Cs...>, Ts...> : func_extra_info<Ts...> {
    static_assert(std::is_same_v<typename func_extra_info<Ts...>::call_guard, void>,
                  "call_guard<> can only be specified once!");
    using call_guard = nanobind::call_guard<Cs...>;
};

template <size_t Nurse, size_t Patient, typename... Ts>
struct func_extra_info<nanobind::keep_alive<Nurse, Patient>, Ts...> : func_extra_info<Ts...> {
    static constexpr bool pre_post_hooks = true;
};

template <typename Policy, typename... Ts>
struct func_extra_info<call_policy<Policy>, Ts...> : func_extra_info<Ts...> {
    static constexpr bool pre_post_hooks = true;
};

template <typename... Ts>
struct func_extra_info<arg_locked, Ts...> : func_extra_info<Ts...> {
    static constexpr size_t nargs_locked = 1 + func_extra_info<Ts...>::nargs_locked;
};

template <typename... Ts>
struct func_extra_info<lock_self, Ts...> : func_extra_info<Ts...> {
    static constexpr size_t nargs_locked = 1 + func_extra_info<Ts...>::nargs_locked;
};

NB_INLINE void process_precall(PyObject **, size_t, detail::cleanup_list *, void *) { }

template <size_t NArgs, typename Policy>
NB_INLINE void
process_precall(PyObject **args, std::integral_constant<size_t, NArgs> nargs,
                detail::cleanup_list *cleanup, call_policy<Policy> *) {
    Policy::precall(args, nargs, cleanup);
}

NB_INLINE void process_postcall(PyObject **, size_t, PyObject *, void *) { }

template <size_t NArgs, size_t Nurse, size_t Patient>
NB_INLINE void
process_postcall(PyObject **args, std::integral_constant<size_t, NArgs>,
                 PyObject *result, nanobind::keep_alive<Nurse, Patient> *) {
    static_assert(Nurse != Patient,
                  "keep_alive with the same argument as both nurse and patient "
                  "doesn't make sense");
    static_assert(Nurse <= NArgs && Patient <= NArgs,
                  "keep_alive template parameters must be in the range "
                  "[0, number of C++ function arguments]");
    keep_alive(Nurse   == 0 ? result : args[Nurse - 1],
               Patient == 0 ? result : args[Patient - 1]);
}

template <size_t NArgs, typename Policy>
NB_INLINE void
process_postcall(PyObject **args, std::integral_constant<size_t, NArgs> nargs,
                 PyObject *&result, call_policy<Policy> *) {
    // result_guard avoids leaking a reference to the return object
    // if postcall throws an exception
    object result_guard = steal(result);
    Policy::postcall(args, nargs, result);
    result_guard.release();
}

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
