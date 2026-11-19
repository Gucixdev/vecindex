# pocket

A phone app that sorts the phone's own files on the phone, with nothing sent anywhere. It is built on
[vecindex](../..) as a library. It has three parts, and this is part one:

1. **cleanup** (here): duplicate and similar photos, the copy to keep, and how much space deleting the rest frees.
2. **document scanner** (next): pages from the camera, cut and flattened, OCR on the device, searchable.
3. **offline transcription** (later): voice to text on the device, searchable next to the photos and scans.

The aim is what the paying apps in these categories get wrong: no weekly subscription, no ads, no account, and one
price. On Android the app asks for no INTERNET permission, so anyone can check on its store page that the photos
cannot leave the phone. The market research behind the choice is the `paying-demand` skill's run of 2026-11-19.

## How the cleanup works

- `pk_photo` turns pixels into a 96-number descriptor and a sharpness score:
  - the descriptor is the 63 lowest frequencies of a 32 x 32 luminance DCT (the DC left out, so brightness does not
    count), plus Cb and Cr means on a 4 x 4 grid, at half weight at most;
  - sharpness is the variance of the Laplacian over the variance of the luminance, at half resolution.
- The descriptors go into a vecindex store, one row per photo. Its fields are `w`, `h`, `sharp`, `bytes` and `time`;
  the row's text is the path.
- `pk_groups` runs one batched search, the 16 nearest of every photo, and joins every pair at or above the
  threshold with union-find:
  - `PK_SAME` 0.97: the same picture (resized, re-encoded, brightened, sent back by a messenger);
  - `PK_SIMILAR` 0.90: the same scene (a burst, a small crop, a blur).
- `pk_keep` picks the photo to keep in a group:
  - copies of one picture (cosine 0.99 or more) are judged by pixels, then file size. A lossy re-encode's block
    edges make it look sharper than its original, so the least compressed copy wins.
  - the best copy of each picture then competes with the other shots on sharpness.

The library takes pixels and allocates nothing. Decoding is the platform's job: on Android, BitmapFactory into
RGBA, then JNI. On the desktop, the CLI reads PNM, and `bin/pocket-scan` decodes with ffmpeg or ImageMagick.

## Use

```sh
./run                                  # builds against ../.., self-test, CLI end to end; ./run asan adds ASan/UBSan
bin/pocket-scan photos.vxi ~/Pictures  # every photo in, each at most 1024 px for the descriptor
.build/pocket dups photos.vxi          # groups, keep/delete, MB freed; "similar" or a cosine for looser; -t for TSV
```

| path | what is there |
|---|---|
| `src/pocket.h`, `src/pocket.c` | the library: descriptor, sharpness, groups, the one to keep |
| `src/pocket_cli.c` | `pocket add` / `pocket dups`, PNM in, mmap only |
| `bin/pocket-scan` | a folder of JPEG/PNG/WebP/HEIC into the store through ffmpeg or ImageMagick |
| `tests/pocket_test.c` | 100 synthetic scenes x 6 copies, the margins printed, groups, keep, reopen, edges |

## Status

Measured on synthetic images only (`./run`, 2026-11-19); not yet on a real phone gallery.

- **Matching:** the lowest cosine of a copy to its original was 1.0000 for half size, 0.9998 for brighter, 0.988 for
  posterized, 1.0000 for blurred and 0.915 for a 4% crop. The highest between two different scenes was 0.691.
  At 0.90, all 600 photos fall into the right 100 groups, with none across scenes.
- **Speed:** about 1250 descriptors a second at 512 x 384 on one core, and 17 ms for a 6-megapixel photo.
- **CLI end to end:** 12 ImageMagick photos, scanned twice, came out in 3 right groups. The half-size copy and the
  re-encode were never kept.
- **Not handled yet:**
  - a rotated or mirrored copy is not matched;
  - between two copies with equal bytes, the brightened one can win over its original (the file time is not used
    yet);
  - a real-gallery check, a JNI layer and the Android app itself are still to do.
