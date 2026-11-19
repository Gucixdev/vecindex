/* ingest.h - documents into a store: Markdown and JSON (and JSON Lines) in C, everything else through them.
 *
 *   vi_ingest_opts o;
 *   vi_ingest_defaults(&o);
 *   o.embed = my_model; o.embed_ctx = model;           your embedding model (NULL: vi_embed_hash, words only)
 *   o.fields = "lang,year,author.name";                front matter / JSON keys kept as fields (filters)
 *   vi_ingest_file(db, "notes/alloc.md", &o, &st);     a file in, or replaced if it was in already
 *
 * A file becomes rows the way vi_chunk and VI_ID say: Markdown by section (a heading starts one; the sections are
 * the pages), each cut into chunks; JSON by element (a top-level array's elements, a JSON Lines file's lines, or the
 * one object), each element's scalars as "path: value" lines of text, cut into chunks. Every row gets the field
 * `path` (the name given) and the fields listed in fields that the file has. The rows of one file are one id range
 * (vi_ingest_range), so a file is replaced or removed whole, and the whole file goes in as one write batch - a crash
 * leaves the old version or the new one, never half. Markdown's headings ("# A > ## B") are put in front of each chunk
 * for the model to embed, not into the stored text, so a context of neighbouring chunks reads as the document does.
 *
 * Other formats, in any language: turn them into Markdown or JSON Lines and pass the text to vi_ingest_text - a PDF
 * through pdftotext (its form feeds between pages are kept: they end chunks), HTML or a .docx into Markdown, a table
 * into JSON Lines - and give the embedding model as the callback (bindings/python: ingest_text with a Python
 * function). The cutting, the ids, the fields, the batch stay here, in one place.
 *
 * Memory: the file is mapped, the working memory is an arena of the call, freed when it returns; no recursion (the
 * JSON parser keeps its own stack, 64 levels deep at most). Built into libvecindex (vecindex.c includes ingest.c). */
#ifndef VECINDEX_INGEST_H
#define VECINDEX_INGEST_H

#include "vecindex.h"

/* An embedding model: text[0, len) into out[0, dim); 0, or non-0 when it failed (the ingest is then abandoned). */
typedef int (*vi_embed_fn)(void *ctx, const char *text, int len, float *out, int dim);

/* The stand-in model: words and word pairs hashed into dim signed buckets, normalised. No model, no download; it
 * finds shared words, not shared meaning - for trying things out, and for word search (vi_search_hybrid) whose
 * vectors do not matter. Not for a Lorentz store (its vectors are not points of the hyperboloid). */
VI_API int vi_embed_hash(void *ctx, const char *text, int len, float *out, int dim);

#define VI_MD    1
#define VI_JSON  2
#define VI_JSONL 3

typedef struct {
    vi_embed_fn embed;      /* default NULL: vi_embed_hash */
    void *embed_ctx;
    int max_bytes;          /* a chunk's bytes at most, as vi_chunk (default 1200) */
    int overlap;            /* as vi_chunk (default 150) */
    const char *fields;     /* keys kept as fields, comma-separated: a front matter key, a JSON path with dots
                             * ("author.name" -> field author_name; array indices left out, the first value wins).
                             * Default NULL: path only. Types: whole numbers int, other numbers float, true/false
                             * int 1/0, the rest strings */
    int headings;           /* Markdown: put the section's headings in front of each chunk for the model (default 1) */
} vi_ingest_opts;

typedef struct {
    int docs;               /* Markdown files 1, JSON elements */
    int sections;           /* Markdown sections, JSON elements: the pages */
    int chunks;             /* rows written */
    int fields;             /* field values set */
    vi_i64 bytes;           /* bytes of text written */
} vi_ingest_stats;

VI_API void vi_ingest_defaults(vi_ingest_opts *o);

/* The ids of a file's rows: [*lo, *hi], from its name (vi_key's). Its documents are 1024 consecutive document numbers
 * from (key >> 31) << 10, so VI_ID(doc, page, chunk) with doc among them. */
VI_API void vi_ingest_range(const char *name, int len, vi_i64 *lo, vi_i64 *hi);

/* path in, by its extension: .md .markdown, .json, .jsonl .ndjson; replacing what was in the store under the same
 * path. st (may be NULL) gets what was done. 0, or -1 (vi_error(): what and where; the store is as it was). */
VI_API int vi_ingest_file(vi_db *db, const char *path, const vi_ingest_opts *o, vi_ingest_stats *st);

/* The same from memory: text[0, len) (len -1: a C string) of kind VI_MD, VI_JSON or VI_JSONL, under name. */
VI_API int vi_ingest_text(vi_db *db, const char *name, const char *text, int len, int kind, const vi_ingest_opts *o,
                          vi_ingest_stats *st);

/* Every row of name out of the store. The number of rows removed, -1. */
VI_API vi_i64 vi_ingest_remove(vi_db *db, const char *name);

#endif
