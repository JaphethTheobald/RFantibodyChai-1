
#ifndef KALIGN_EXPORT_H
#define KALIGN_EXPORT_H

#ifdef KALIGN_STATIC_DEFINE
#  define KALIGN_EXPORT
#  define KALIGN_NO_EXPORT
#else
#  ifndef KALIGN_EXPORT
#    ifdef kalign_EXPORTS
        /* We are building this library */
#      define KALIGN_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define KALIGN_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef KALIGN_NO_EXPORT
#    define KALIGN_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef KALIGN_DEPRECATED
#  define KALIGN_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef KALIGN_DEPRECATED_EXPORT
#  define KALIGN_DEPRECATED_EXPORT KALIGN_EXPORT KALIGN_DEPRECATED
#endif

#ifndef KALIGN_DEPRECATED_NO_EXPORT
#  define KALIGN_DEPRECATED_NO_EXPORT KALIGN_NO_EXPORT KALIGN_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef KALIGN_NO_DEPRECATED
#    define KALIGN_NO_DEPRECATED
#  endif
#endif

#endif /* KALIGN_EXPORT_H */
