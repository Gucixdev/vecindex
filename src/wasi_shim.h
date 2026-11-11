/* wasi_shim.h - what WASI (WebAssembly, wasm32-wasip1) lacks, for vecindex built to WASM (bindings/js). There are no
 * processes and no signals: a background snapshot cannot fork, so it is refused (ENOSYS) and the caller saves
 * instead. Threads are wasi-libc's stubs in a build without wasm threads: pthread_create fails, so the store's own
 * indexing and checking threads never start and the work happens on the calling thread. Included by store.c only
 * when __wasi__. */
#ifndef VECINDEX_WASI_SHIM_H
#define VECINDEX_WASI_SHIM_H
#include <errno.h>
#include <sys/types.h>
#define WNOHANG 1
#define WIFEXITED(s) 0
#define WEXITSTATUS(s) 0
#define WIFSIGNALED(s) 0
static pid_t vi_wasi_fork_(void) { errno = ENOSYS; return -1; }
static pid_t vi_wasi_waitpid_(pid_t p, int *st, int o) { (void)p; (void)st; (void)o; errno = ECHILD; return -1; }
#define fork vi_wasi_fork_
#define waitpid vi_wasi_waitpid_

/* mkstemp: the XXXXXX made of a counter and the clock, opened with O_EXCL until a name is free */
#include <fcntl.h>
#include <string.h>
#include <time.h>
static int vi_wasi_mkstemp_(char *t)
{
    static unsigned n;
    size_t l = strlen(t);
    int i, fd = -1, tries;
    if (l < 6 || strcmp(t + l - 6, "XXXXXX")) { errno = EINVAL; return -1; }
    for (tries = 0; tries < 100 && fd < 0; ++tries) {
        unsigned x = (unsigned)time((time_t *)0) * 2654435761u + ++n * 40503u;
        for (i = 0; i < 6; ++i) { t[l - 6 + i] = "abcdefghijklmnopqrstuvwxyz0123456789"[x % 36u]; x /= 36u; }
        fd = open(t, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd < 0 && errno != EEXIST) return -1;
    }
    return fd;
}
#define mkstemp vi_wasi_mkstemp_

/* the memory advice and pinning of a mapping: no such thing in WebAssembly's one linear memory */
static int vi_wasi_madvise_(void *p, size_t n, int a) { (void)p; (void)n; (void)a; return 0; }
static int vi_wasi_mlock_(const void *p, size_t n) { (void)p; (void)n; errno = ENOSYS; return -1; }
static int vi_wasi_munlock_(const void *p, size_t n) { (void)p; (void)n; return 0; }
#define madvise vi_wasi_madvise_
/* wasi-libc's emulated mmap takes anonymous and private file mappings only, and no flags beside: a reservation is
 * plain memory there anyway */
#include <sys/mman.h>
#undef MAP_NORESERVE
#define MAP_NORESERVE 0
#define mlock vi_wasi_mlock_
#define munlock vi_wasi_munlock_
#endif
