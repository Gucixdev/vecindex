/* config.h - every setting of vecindex that is fixed when it is compiled, in one place. No config file and no
 * parser: edit a value here (or pass -DNAME=value to the compiler) and rebuild. Each has a default that suits a
 * machine it knows nothing about; what can change while a store is open has a call instead (vi_set_recall,
 * vi_set_layout, ...). store.c includes this first. */
#ifndef VECINDEX_CONFIG_H
#define VECINDEX_CONFIG_H

/* --- memory ------------------------------------------------------------------------------------------------ */

/* RAM the store may count on, in MB. 0: what the machine has (or its cgroup allows), found when the store starts.
 * The store keeps to half of it for what must sit in RAM: an index whose codes are larger loads cold (pages in
 * only the clusters a query probes), a file larger is checked lazily. With a limit, rows also spill to a file
 * sooner (a quarter of the limit, 256 MB at most). Not a hard cap: the OS page cache and the rows of a query in
 * flight come on top. */
#ifndef VI_RAM_LIMIT_MB
#define VI_RAM_LIMIT_MB 0
#endif

/* VRAM the optional GPU scan may take, in MB. 0: half of what the device reports. Read by the Vulkan path only
 * (not built yet: the setting is here so its limit is in the same place as the RAM one). */
#ifndef VI_VRAM_LIMIT_MB
#define VI_VRAM_LIMIT_MB 0
#endif

#if VI_RAM_LIMIT_MB > 0 && VI_RAM_LIMIT_MB < 1024 && !defined(STORE_SPILL_MIN)
#define STORE_SPILL_MIN ((i64)(VI_RAM_LIMIT_MB / 4) << 20)
#endif

/* --- search ------------------------------------------------------------------------------------------------ */

/* recall@10 a store opened without VI_MANUAL keeps (the tuner picks the fewest probes that reach it). Lower is
 * faster: 0.90 is about 1.6 times the queries a second of 0.95 on GloVe-100. 0: no tuning. */
#ifndef VI_RECALL_DEFAULT
#define VI_RECALL_DEFAULT 0.90f
#endif

/* --- the index --------------------------------------------------------------------------------------------- */

/* rows from which the index is clustered (below: one flat scan, exact-ish and fast enough) */
#ifndef STORE_IVF_MIN
#define STORE_IVF_MIN 600000
#endif

/* no cluster is planned smaller than this many rows (the count itself: 4 sqrt(rows), lean4/Vecindex/Config.lean) */
#ifndef STORE_IVF_ROWS
#define STORE_IVF_ROWS 256
#endif

/* live mode splits a cluster in place once it holds this many times the store's cluster size */
#ifndef STORE_SPLIT_X
#define STORE_SPLIT_X 2.0
#endif

/* clusters from which a query picks its probes through a second level of top cells (tq_top.c) instead of
 * comparing itself with every centroid: 16384 clusters is about 16M rows */
#ifndef VI_TOP_MIN
#define VI_TOP_MIN 16384
#endif

#endif
