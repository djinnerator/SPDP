import struct
import numpy as np
import torch
from typing import Union


def LNVs2_compress(words):
    """
    Word-granularity LNVs2 component.
    Subtracts the second-previous value from the current value using arithmetic subtraction and emits the residual[cite: 2].
    """
    out = []
    prev2, prev1 = 0, 0
    for w in words:
        # Arithmetic subtraction of the second-previous value[cite: 2]
        residual = (w - prev2) & 0xFFFFFFFF
        out.append(residual)
        prev2 = prev1
        prev1 = w
    return out


def LNVs2_decompress(words):
    """Inverse of LNVs2 component."""
    out = []
    prev2, prev1 = 0, 0
    for residual in words:
        w = (residual + prev2) & 0xFFFFFFFF
        out.append(w)
        prev2 = prev1
        prev1 = w
    return out


def cut_to_bytes(words):
    """
    The Cut component (|).
    Converts a sequence of words into a sequence of bytes[cite: 2].
    """
    out = bytearray()
    for w in words:
        out.extend(struct.pack('<I', w))
    return out


def cut_to_words(byte_seq):
    """Inverse of the Cut component."""
    words = []
    for i in range(0, len(byte_seq), 4):
        words.append(struct.unpack('<I', byte_seq[i:i + 4])[0])
    return words


def DIM8_compress(byte_seq):
    """
    DIM8 component.
    Groups the values according to a dimensionality of 8, grouping most significant bytes together, followed by the second most significant bytes, etc.[cite: 2].
    """
    out = bytearray(len(byte_seq))
    wpos = 0
    for d in range(8):
        for rpos in range(d, len(byte_seq), 8):
            out[wpos] = byte_seq[rpos]
            wpos += 1
    return out


def DIM8_decompress(byte_seq):
    """Inverse of the DIM8 component."""
    out = bytearray(len(byte_seq))
    rpos = 0
    for d in range(8):
        for wpos in range(d, len(byte_seq), 8):
            out[wpos] = byte_seq[rpos]
            rpos += 1
    return out


def LNVs1_compress(byte_seq):
    """
    Byte-granularity LNVs1 component.
    Subtracts the last 1st value from the current value (arithmetic subtraction)[cite: 2].
    """
    out = bytearray()
    prev = 0
    for b in byte_seq:
        out.append((b - prev) & 0xFF)
        prev = b
    return out


def LNVs1_decompress(byte_seq):
    """Inverse of the LNVs1 component."""
    out = bytearray()
    prev = 0
    for b in byte_seq:
        val = (b + prev) & 0xFF
        out.append(val)
        prev = val
    return out


def LZa6_compress(byte_seq):
    """
    LZa6 component (LZ77 variant).
    Uses a 32768-entry hash table to find recent prior occurrences[cite: 2].
    Checks if n=6 values immediately preceding match the 6 values before the current location[cite: 2].
    If they match, counts subsequent matches and emits the length; otherwise emits the current value[cite: 2].
    """
    out = bytearray()
    # 32768-entry hash table to store the most recent prior occurrences[cite: 2]
    hash_table = [0] * 32768

    rpos = 0
    length = len(byte_seq)

    while rpos < length:
        val = byte_seq[rpos]
        # Hash lookup for current value (using a simple hash function fitting the 32768 table)
        h = (val * 2654435761) % 32768
        lpos = hash_table[h]

        match_found = False
        # Check if the 6 values immediately preceding match[cite: 2]
        if lpos >= 6 and rpos >= 6:
            if byte_seq[lpos - 6:lpos] == byte_seq[rpos - 6:rpos]:
                match_found = True

        if match_found:
            # Count how many values following the current value match the values after the location[cite: 2]
            match_len = 0
            while rpos < length and lpos < length and byte_seq[rpos] == byte_seq[lpos] and match_len < 255:
                # Update hash table as we advance
                curr_val = byte_seq[rpos]
                curr_h = (curr_val * 2654435761) % 32768
                hash_table[curr_h] = rpos

                match_len += 1
                rpos += 1
                lpos += 1

            # The length of the matching substring is emitted[cite: 2]
            out.append(match_len)
        else:
            # If they do not match, only the current value is emitted[cite: 2]
            out.append(val)
            hash_table[h] = rpos
            rpos += 1

    return out


def LZa6_decompress(byte_seq, original_length):
    """
    Inverse of the LZa6 component.
    Reconstructs the stream using the emitted literal values and match lengths.
    """
    out = bytearray()
    hash_table = [0] * 32768

    rpos = 0
    while len(out) < original_length:
        wpos = len(out)

        # We need the most recent byte to determine the hash if we are simulating the lookback
        if wpos > 0:
            val = out[-1]
            h = (val * 2654435761) % 32768
            lpos = hash_table[h]
        else:
            lpos = 0

        match_found = False
        if lpos >= 6 and wpos >= 6:
            if out[lpos - 6:lpos] == out[wpos - 6:wpos]:
                match_found = True

        if match_found:
            match_len = byte_seq[rpos]
            rpos += 1
            for _ in range(match_len):
                val = out[lpos]
                out.append(val)

                curr_h = (val * 2654435761) % 32768
                hash_table[curr_h] = len(out) - 1
                lpos += 1
        else:
            val = byte_seq[rpos]
            out.append(val)
            rpos += 1

            curr_h = (val * 2654435761) % 32768
            hash_table[curr_h] = len(out) - 1

    return out


def spdp_compress(data: Union[torch.Tensor, np.ndarray, bytearray, bytes]):
    """Full SPDP Compression Pipeline[cite: 2]"""
    if isinstance(data, (torch.Tensor, np.ndarray)):
        data_bytes = bytearray(memoryview(data))
    else: data_bytes = data
    # Pad to 4-byte boundaries if necessary (words requirement)
    padding = (4 - (len(data_bytes) % 4)) % 4
    data_bytes += b'\x00' * padding

    words = cut_to_words(data_bytes)

    # 1. LNVs2[cite: 2]
    stage1 = LNVs2_compress(words)
    # 2. Cut[cite: 2]
    stage2 = cut_to_bytes(stage1)
    # 3. DIM8[cite: 2]
    stage3 = DIM8_compress(stage2)
    # 4. LNVs1[cite: 2]
    stage4 = LNVs1_compress(stage3)
    # 5. LZa6[cite: 2]
    compressed = LZa6_compress(stage4)

    return compressed, data.shape, len(data_bytes), type(data), data.dtype


def spdp_decompress(
        compressed_bytes: bytearray,
        original_shape: tuple,
        original_length: int,
        original_type: Union[torch.Tensor, np.ndarray, bytearray, bytes],
        original_dtype,
):
    """Full SPDP Decompression Pipeline[cite: 2]"""
    # 1. Inverse LZa6[cite: 2]
    stage4 = LZa6_decompress(compressed_bytes, original_length)
    # 2. Inverse LNVs1[cite: 2]
    stage3 = LNVs1_decompress(stage4)
    # 3. Inverse DIM8[cite: 2]
    stage2 = DIM8_decompress(stage3)
    # 4. Inverse Cut[cite: 2]
    stage1 = cut_to_words(stage2)
    # 5. Inverse LNVs2[cite: 2]
    words = LNVs2_decompress(stage1)

    decompressed_bytes = cut_to_bytes(words)
    decompressed_bytes = decompressed_bytes[:original_length]
    if original_type == np.ndarray:
        return np.frombuffer(decompressed_bytes, dtype=original_dtype).reshape(original_shape)
    elif original_type == torch.Tensor:
        return torch.frombuffer(decompressed_bytes, dtype=original_dtype).reshape(original_shape)
    else:  # original_type == bytearray or original_type == bytes
        return decompressed_bytes
    # return decompressed_bytes[:original_length]