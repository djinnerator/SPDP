"""
SPDP Floating-Point Compressor (Python Implementation)
Compatible with SPDP v1.1 reference implementation.

Original Algorithm & C Reference:
    Steven Claggett and Martin Burtscher.
    "SPDP: An Automatically Synthesized Lossless Compression Algorithm for
    Floating-Point Data." Proceedings of the 2018 Data Compression Conference,
    pp. 337-346. March 2018.
    Copyright (c) 2015-2020, Texas State University.
"""

from __future__ import annotations

import array
import io
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Union

# Algorithm Constants matching SPDP_11.c
BUFFER_SIZE: int = 1 << 23       # 8 MB per block chunk
MAX_TABLE_SIZE: int = 1 << 18    # 262,144 prediction table entries
DEFAULT_LEVEL: int = 6

# Unsigned 32-bit integer array typecode (word_t in C)
_WORD_TYPECODE: str = "I" if array.array("I").itemsize == 4 else "L"
_IS_BIG_ENDIAN: bool = sys.byteorder == "big"


# ---------------------------------------------------------------------------
# Low-level Block Compression & Decompression
# ---------------------------------------------------------------------------

def compress_block(data: bytes | bytearray | memoryview, level: int = DEFAULT_LEVEL) -> bytes:
    """
    Compress a single block of bytes using the SPDP algorithm.

    Args:
        data: Raw uncompressed bytes (ideal for 32-bit float or 64-bit double sequences).
        level: Compression level (0 to 9). Higher values use larger history tables.

    Returns:
        The compressed block bytes (without framing headers).
    """
    length = len(data)
    if length == 0:
        return b""

    level = max(0, min(9, int(level)))
    buf1 = bytearray(data)

    # -----------------------------------------------------------------------
    # Stage 1: 32-bit 2nd-order differentiation across words
    # -----------------------------------------------------------------------
    len_words = length // 4
    if len_words > 0:
        words = array.array(_WORD_TYPECODE)
        words.frombytes(buf1[: len_words * 4])
        if _IS_BIG_ENDIAN:
            words.byteswap()

        prev2 = 0
        prev1 = 0
        for pos in range(len_words):
            curr = words[pos]
            words[pos] = (curr - prev2) & 0xFFFFFFFF
            prev2 = prev1
            prev1 = curr

        if _IS_BIG_ENDIAN:
            words.byteswap()

        buf2 = bytearray(words.tobytes())
        buf2.extend(buf1[len_words * 4 : length])
    else:
        buf2 = bytearray(buf1)

    # -----------------------------------------------------------------------
    # Stage 2: 8-way byte interleaving and 1st-order byte delta
    # -----------------------------------------------------------------------
    buf1 = bytearray(length)
    prev = 0
    wpos = 0
    for d in range(8):
        for rpos in range(d, length, 8):
            curr = buf2[rpos]
            buf1[wpos] = (curr - prev) & 0xFF
            prev = curr
            wpos += 1

    # -----------------------------------------------------------------------
    # Stage 3: History table prediction & run-length encoding
    # -----------------------------------------------------------------------
    predtabsize = 1 << (level + 9)
    if predtabsize > MAX_TABLE_SIZE:
        predtabsize = MAX_TABLE_SIZE
    predtabsizem1 = predtabsize - 1

    lastpos = [0] * predtabsize
    out = bytearray()
    rpos = 0
    hist = 0

    while rpos < length:
        val = buf1[rpos]
        lpos = lastpos[hist]
        if lpos >= 6:
            # Check 6-byte match with early exit on most recent byte
            if (
                buf1[lpos - 1] == buf1[rpos - 1]
                and buf1[lpos - 2] == buf1[rpos - 2]
                and buf1[lpos - 3] == buf1[rpos - 3]
                and buf1[lpos - 4] == buf1[rpos - 4]
                and buf1[lpos - 5] == buf1[rpos - 5]
                and buf1[lpos - 6] == buf1[rpos - 6]
            ):
                cnt = 0
                while val == buf1[lpos] and cnt < 255 and rpos < (length - 1):
                    lastpos[hist] = rpos
                    hist = ((hist << 2) ^ val) & predtabsizem1
                    rpos += 1
                    lpos += 1
                    cnt += 1
                    val = buf1[rpos]
                out.append(cnt)

        out.append(val)
        lastpos[hist] = rpos
        hist = ((hist << 2) ^ val) & predtabsizem1
        rpos += 1

    return bytes(out)


def decompress_block(compressed_data: bytes | bytearray | memoryview, level: int = DEFAULT_LEVEL) -> bytes:
    """
    Decompress a single block of SPDP-compressed bytes.

    Args:
        compressed_data: Raw compressed block bytes (without framing headers).
        level: Compression level (0 to 9) that was used during compression.

    Returns:
        The uncompressed block bytes.
    """
    csize = len(compressed_data)
    if csize == 0:
        return b""

    level = max(0, min(9, int(level)))
    buf2 = compressed_data

    # -----------------------------------------------------------------------
    # Stage 3 Inverse: LZ/RLE reconstruction
    # -----------------------------------------------------------------------
    predtabsize = 1 << (level + 9)
    if predtabsize > MAX_TABLE_SIZE:
        predtabsize = MAX_TABLE_SIZE
    predtabsizem1 = predtabsize - 1

    lastpos = [0] * predtabsize
    buf1 = bytearray()
    rpos = 0
    hist = 0

    while rpos < csize:
        lpos = lastpos[hist]
        wpos = len(buf1)
        if lpos >= 6:
            if (
                buf1[lpos - 1] == buf1[wpos - 1]
                and buf1[lpos - 2] == buf1[wpos - 2]
                and buf1[lpos - 3] == buf1[wpos - 3]
                and buf1[lpos - 4] == buf1[wpos - 4]
                and buf1[lpos - 5] == buf1[wpos - 5]
                and buf1[lpos - 6] == buf1[wpos - 6]
            ):
                cnt = buf2[rpos]
                rpos += 1
                for _ in range(cnt):
                    val = buf1[lpos]
                    buf1.append(val)
                    lastpos[hist] = wpos
                    hist = ((hist << 2) ^ val) & predtabsizem1
                    wpos += 1
                    lpos += 1

        val = buf2[rpos]
        buf1.append(val)
        lastpos[hist] = wpos
        hist = ((hist << 2) ^ val) & predtabsizem1
        rpos += 1

    usize = len(buf1)

    # -----------------------------------------------------------------------
    # Stage 2 Inverse: 8-way byte de-interleaving and cumulative sum
    # -----------------------------------------------------------------------
    stage2_out = bytearray(usize)
    val = 0
    rpos = 0
    for d in range(8):
        for wpos in range(d, usize, 8):
            val = (val + buf1[rpos]) & 0xFF
            stage2_out[wpos] = val
            rpos += 1

    # -----------------------------------------------------------------------
    # Stage 1 Inverse: 32-bit 2nd-order cumulative sum across words
    # -----------------------------------------------------------------------
    len_words = usize // 4
    if len_words > 0:
        words = array.array(_WORD_TYPECODE)
        words.frombytes(stage2_out[: len_words * 4])
        if _IS_BIG_ENDIAN:
            words.byteswap()

        prev2 = 0
        prev1 = 0
        for pos in range(len_words):
            curr = (words[pos] + prev2) & 0xFFFFFFFF
            words[pos] = curr
            prev2 = prev1
            prev1 = curr

        if _IS_BIG_ENDIAN:
            words.byteswap()

        out = bytearray(words.tobytes())
        out.extend(stage2_out[len_words * 4 : usize])
        return bytes(out)

    return bytes(stage2_out)


# ---------------------------------------------------------------------------
# Stream API (Binary-compatible with SPDP_11.c file format)
# ---------------------------------------------------------------------------

def compress_stream(in_stream: BinaryIO, out_stream: BinaryIO, level: int = DEFAULT_LEVEL) -> None:
    """
    Compress an input binary stream to an output binary stream in SPDP format.
    Format:
        1 byte:  compression level (0-9)
        Repeated blocks:
            4 bytes: uncompressed block size (int32, little-endian)
            4 bytes: compressed block size (int32, little-endian)
            N bytes: compressed block data
    """
    level = max(0, min(9, int(level)))
    out_stream.write(struct.pack("B", level))

    while True:
        chunk = in_stream.read(BUFFER_SIZE)
        if not chunk:
            break
        length = len(chunk)
        compressed_chunk = compress_block(chunk, level)
        csize = len(compressed_chunk)
        out_stream.write(struct.pack("<ii", length, csize))
        out_stream.write(compressed_chunk)


def decompress_stream(in_stream: BinaryIO, out_stream: BinaryIO) -> None:
    """
    Decompress an SPDP binary stream to an uncompressed output binary stream.
    """
    header = in_stream.read(1)
    if not header:
        return

    level = struct.unpack("B", header)[0]
    if level > 9:
        raise ValueError(f"Invalid SPDP file format or compression level: {level}")

    while True:
        header_bytes = in_stream.read(8)
        if not header_bytes:
            break
        if len(header_bytes) < 8:
            raise EOFError("Truncated SPDP block header.")

        length, csize = struct.unpack("<ii", header_bytes)
        compressed_chunk = in_stream.read(csize)
        if len(compressed_chunk) < csize:
            raise EOFError("Truncated SPDP compressed payload.")

        decompressed_chunk = decompress_block(compressed_chunk, level)
        if len(decompressed_chunk) != length:
            raise ValueError(
                f"Decompressed block size mismatch: expected {length}, got {len(decompressed_chunk)}"
            )
        out_stream.write(decompressed_chunk)


# ---------------------------------------------------------------------------
# High-Level Byte API
# ---------------------------------------------------------------------------

def compress(data: bytes | bytearray | memoryview, level: int = DEFAULT_LEVEL) -> bytes:
    """
    Compress raw bytes into a complete SPDP-formatted byte string.
    """
    in_buf = io.BytesIO(data)
    out_buf = io.BytesIO()
    compress_stream(in_buf, out_buf, level)
    return out_buf.getvalue()


def decompress(data: bytes | bytearray | memoryview) -> bytes:
    """
    Decompress an SPDP-formatted byte string back into the original bytes.
    """
    in_buf = io.BytesIO(data)
    out_buf = io.BytesIO()
    decompress_stream(in_buf, out_buf)
    return out_buf.getvalue()


# ---------------------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------------------

def compress_file(
    input_path: Union[str, Path], output_path: Union[str, Path], level: int = DEFAULT_LEVEL
) -> None:
    """Compress a file on disk."""
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        compress_stream(fin, fout, level)


def decompress_file(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Decompress a file on disk."""
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        decompress_stream(fin, fout)


# ---------------------------------------------------------------------------
# Command-Line Interface (Matches SPDP C CLI)
# ---------------------------------------------------------------------------

def _main() -> None:
    args = sys.argv[1:]

    # Supports original C syntax:
    #   Compression:   python spdp.py level < uncompressed > compressed
    #   Decompression: python spdp.py < compressed > decompressed
    if len(args) == 1 and args[0].isdigit():
        level = int(args[0])
        compress_stream(sys.stdin.buffer, sys.stdout.buffer, level)
    elif len(args) == 0:
        decompress_stream(sys.stdin.buffer, sys.stdout.buffer)
    else:
        sys.stderr.write("SPDP Floating-Point Compressor v1.1 (Python port)\n")
        sys.stderr.write("Usage:\n")
        sys.stderr.write("  Compression:   python spdp.py [level 0-9] < uncompressed > compressed\n")
        sys.stderr.write("  Decompression: python spdp.py < compressed > decompressed\n")
        sys.exit(1)


if __name__ == "__main__":
    _main()