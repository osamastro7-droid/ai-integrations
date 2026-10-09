"""A payload codec for testing converter composition across Temporal boundaries."""

import zlib
from collections.abc import Sequence

from temporalio.api.common.v1 import Payload
from temporalio.converter import PayloadCodec


class CompressionCodec(PayloadCodec):
    """Wrap each serialized payload in a compressed payload with its own encoding."""

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload(
                metadata={"encoding": b"binary/zlib"},
                data=zlib.compress(payload.SerializeToString()),
            )
            for payload in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return [
            Payload.FromString(zlib.decompress(payload.data)) for payload in payloads
        ]
