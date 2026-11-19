# Compressed texts - the design, before any code

Asked 2026-11-19: the documents stay apart from the vectors and take little space, and "nothing may fall apart":
compression goes through Reed-Solomon like every other byte of the file. This is the plan the code will follow; the
tests listed at the end are what "done" means. Nothing here is built yet.

## What it must keep

1. **Every compressed byte is under the file's block digests and Reed-Solomon parity.** Compression happens first,
   when the file is written (`store_write_file_`); the digests and parity are computed after, over the bytes as they
   lie in the file (`ssum_write_file_`). Damage within what the parity rebuilds (2 blocks of every 32 by default)
   comes back bit for bit before anything is decompressed, so a text never sees it.
2. **Damage beyond that loses one text, never more.** Each text is compressed on its own (a frame), so a back-reference
   can only reach into its own frame and the dictionary - never into a neighbour. A frame that fails to decode, or
   whose decoded bytes fail the text's CRC, is that text alone: not served, reported by `vi_integrity`, and the
   store, its vectors and every other text open and work as before.
3. **The dictionary is never a single point of failure.** It is small (32 to 64 KB) and stored twice, each copy with
   its own CRC-32C and BLAKE3 digest, far apart in the file and both under the parity. A load takes the first copy
   that checks; if both are damaged beyond repair, the compressed texts are reported damaged and the store still
   opens (vectors, fields, labels, raw texts all work).
4. **The CRC stays on the raw text.** `tck` is computed on the text as it came and checked after decoding, so it
   catches damaged bytes and a wrong decoder alike.
5. **The decoder never steps outside its buffers.** It knows the raw length; every literal run is checked against the
   end of the frame, every match against the bytes already produced plus the dictionary, and the output against the
   raw length. Any violation is "damaged", never a crash or a read past the end. It gets a fuzz target
   (`tests/fuzz_text.c`) over random frames, and a Lean statement of the bounds (the index of every read and write
   inside its buffer) beside the other proofs.
6. **Nothing is worse than raw.** A text that does not shrink (already compressed, very short, random) is stored raw,
   with its frame saying so; so is every text in a store that asked for no compression.
7. **The format is a new flag**, so a build before it refuses the file instead of misreading it (format.md's rule).

## The layout

- The text heap becomes a heap of frames: `kind` (u8: 0 raw, 1 compressed with the dictionary), the frame's length
  (u24; the raw length is in `txo` as today), then the bytes. `txo` keeps `offset | raw length << 40`.
- The dictionary: after the heap, `length u32, crc u32, blake3[32], bytes`, and the same again at the end of the data
  region, before the digests. A file flag (next free bit) says the heap holds frames.
- In memory, texts added since the last save stay raw (writes cost nothing extra); a save compresses them; a loaded
  file's frames are decoded on read - `vi_get_text`, `vi_context`, the BM25 build - into the caller's buffer or a
  scratch one (`scratch.h`, no malloc), stopping as soon as the asked-for bytes are there.

## The coder

LZ77 in the LZ4 block style: a token (4 bits of literal length, 4 bits of match length), literals, a 16-bit offset;
a 64 KB window that starts in the dictionary, so a short chunk can point into it. Our own code in C89, no library;
decoding is a few copies a token. The dictionary is trained at save time from a sample of the store's texts - the
substrings that recur in most of them, most frequent last (the cheapest offsets) - and rebuilt only when the texts
have changed by a set share since the last one. Measured before it is the default: ratio and decode MB/s on real
corpora (this repo's docs, a Wikipedia sample, source code), against raw and against storing the frame raw.

## Done means

- round trip of every text, every kind, with and without dictionary; incompressible texts stored raw
- a block of compressed bytes damaged within the parity's reach: every text identical after the load
- damage beyond it inside one frame: that text reported, every other text, vector and field intact
- the first dictionary copy damaged: the second used, nothing reported lost; both damaged: the compressed texts
  reported, the store open, raw texts and vectors working
- the fuzz target clean under ASan and UBSan; the bounds proved
- a file of the build before it refused, not misread; `./run` all green on his hardware
