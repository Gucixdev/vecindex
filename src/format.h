/* format.h - the file format's constants and layout in one place. What stays compatible, and how a version is
 * added, is docs/format.md; the code that writes and reads is store_disk.c (the file and its journal) and
 * store_sum.c (digests and parity). Everything in a file is little-endian (core.c swaps on a big-endian host), and
 * every section starts on a 64-byte boundary, so a load maps the file and points into it.
 *
 * NAME.vxi, version 9 and 10:
 *   header     4 x i32: magic 'VIX4', version, dim, live rows
 *   extension  4 x i32: flags (below), metric (0 cosine, 1 dot, 2 L2), the stamp's low and high word
 *   goal       8 x i32: the retired engine's build goal; only its seed is read (the index's rotation)
 *   columns    vectors (f32, or int8 with flag 1: then their scales), ids (i64), tags (i64), row checks,
 *              norms (metric != 0), texts (flag 2: heap length, offset | length << 40 a row, the heap),
 *              labels (4), numbers (8), namespaces (16), text CRCs (32)
 *   kind       u8: 0 no index (a load builds it), 2 the scan index follows (1: the retired engine's, read as 0)
 *   scan index 8 x i32: dim, dpad, rows, blocks, clusters, probes, bit 0 slot rows / bit 1 top cells,
 *              code bits (0: 4, 5: 4 then the 1-bit cascade); the rotation's two sign vectors and permutation;
 *              with clusters: centroids, cluster starts; with top cells (flag 64): their count, the cells, their
 *              starts, the clusters by cell; slot rows; the codes; the cascade's codes
 *   sums       block digests, Reed-Solomon parity, two copies of the table and its trailer ('VXS1', store_sum.c)
 *
 * NAME.vxi.j, the journal: 'VXJ3', dim, stamp and their CRC-32C, then frames 'G' len crc records. */
#ifndef VECINDEX_FORMAT_H
#define VECINDEX_FORMAT_H

#define STORE_MAGIC   0x34584956u   /* 'VIX4' - v4 = goal gains max_candidates */
#define STORE_VERSION 5             /* v5: 64-byte aligned sections, the scan index saved too, loaded by mmap */
#define STORE_VERSION_I8 6          /* v6: v5 + a 16-byte extension after the header (int8 rows) and their scales */
#define STORE_VERSION7 7            /* v7: the extension on every file - flags (bit 0: int8 rows), 0, and the 64-bit
                                     * stamp its journal carries too */
#define STORE_VERSION8 8            /* v8: v7 + texts (flags bit 1): after the meta column, the heap's length (u64),
                                     * each row's text offset | length << 40 (u64), then the heap; written only by a
                                     * store that holds texts */
#define STORE_VERSION9 9            /* v9: the v7/v8 layout (texts: flags bit 1), then block digests, Reed-Solomon
                                     * parity and two copies of the table of digests and its trailer (store_sum.c) */
#define STORE_VERSION10 10          /* v10: v9 with features a v9 reader does not know (flags 8 numbers, 16 namespaces, 32
                                     * text CRCs) and would misread: so that it refuses the file. A store without them is
                                     * still written as v9, which every reader since v9 opens (docs/format.md) */
#define STORE_FLAGS_V9 7            /* the flags a v9 reader knows: int8 rows, texts, labels */
#define STORE_F_TOP 64              /* flag 64: the scan index carries its top cells (tq_top.c); written only when it has them */
#define STORE_FLAGS_KNOWN 127       /* every flag this build reads: a file with another is from a newer build, refused */
#define STORE_V9UP(v) ((v) == STORE_VERSION9 || (v) == STORE_VERSION10)   /* the v9 layout, with its digests and parity */

/* the journal's magics: bound to its file by size (1), by the stamp (2), and with a CRC'd header and frames (3) */
#define STORE_JMAGIC  0x314a5856u
#define STORE_JMAGIC2 0x324a5856u
#define STORE_JMAGIC3 0x334a5856u

/* the table of digests and parity at the end of a v9+ file */
#define SSUM_MAGIC 0x31535856u      /* 'VXS1' */

#endif
