/* fuzz_main.h - main() for a fuzz target built without libFuzzer: LLVMFuzzerTestOneInput once on each file named, or on
 * each file in each directory named, so ./run can replay the seeds and the inputs that once failed with gcc and its
 * sanitizers alone. A target includes it at its end; -DFUZZ_MAIN turns it on. */
#ifdef FUZZ_MAIN
#include <dirent.h>
#include <sys/stat.h>

static int fz_replay_(const char *path)
{
    FILE *f = fopen(path, "rb");
    unsigned char *b;
    long n;
    if (!f) { fprintf(stderr, "fuzz replay: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    b = (unsigned char *)malloc(n > 0 ? (size_t)n : 1);
    if (!b || (n > 0 && fread(b, 1, (size_t)n, f) != (size_t)n)) { fclose(f); free(b); return -1; }
    fclose(f);
    LLVMFuzzerTestOneInput(b, (size_t)n);
    free(b);
    return 0;
}

int main(int argc, char **argv)
{
    int i, n = 0, bad = 0;
    LLVMFuzzerInitialize(&argc, &argv);
    for (i = 1; i < argc; ++i) {
        struct stat st;
        if (stat(argv[i], &st) == 0 && S_ISDIR(st.st_mode)) {
            DIR *d = opendir(argv[i]);
            struct dirent *e;
            char p[4096];
            while (d && (e = readdir(d))) {
                if (e->d_name[0] == '.') continue;
                snprintf(p, sizeof(p), "%s/%s", argv[i], e->d_name);
                if (stat(p, &st) == 0 && S_ISREG(st.st_mode)) { if (fz_replay_(p)) ++bad; else ++n; }
            }
            if (d) closedir(d);
        } else if (fz_replay_(argv[i])) ++bad; else ++n;
    }
    printf("  ok   %d inputs replayed%s\n", n, bad ? ", some unreadable" : "");
    return bad ? 1 : 0;
}
#endif
