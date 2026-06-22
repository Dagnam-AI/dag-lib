"""Descriptor format decoder registry."""

from __future__ import annotations

from dagnam.data.loaders.system.decoders.array import ArrayDecoder
from dagnam.data.loaders.system.decoders.audio_folder import AudioFolderDecoder
from dagnam.data.loaders.system.decoders.base import DecodeError, FormatDecoder
from dagnam.data.loaders.system.decoders.image_folder import ImageFolderDecoder
from dagnam.data.loaders.system.decoders.image_mask_folder import ImageMaskFolderDecoder
from dagnam.data.loaders.system.decoders.tabular import TabularDecoder
from dagnam.data.loaders.system.decoders.text import TextDecoder

DECODERS: dict[str, FormatDecoder] = {
    "array": ArrayDecoder(),
    "audio_folder": AudioFolderDecoder(),
    "image_folder": ImageFolderDecoder(),
    "image_mask_folder": ImageMaskFolderDecoder(),
    "tabular": TabularDecoder(),
    "text": TextDecoder(),
}


def get_decoder(format_name: str) -> FormatDecoder:
    """Return the decoder for a descriptor format."""
    decoder = DECODERS.get(format_name)
    if decoder is None:
        raise DecodeError(f"unknown format {format_name!r}; supported: {sorted(DECODERS)}")
    return decoder
