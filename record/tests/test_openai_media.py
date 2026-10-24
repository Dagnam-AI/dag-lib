"""Media in OpenAI responses (generated images, audio) is replaced by its size, never recorded."""

from __future__ import annotations

from typing import Any, Dict, List

from _mock_server import Event, Reply
from _openai_bodies import (
    CHAT_PATH,
    MODEL,
    RESPONSES_PATH,
    chat_body,
    response_body,
    response_events,
)
import pytest

BIG = "QUJD" * 300_000  # 1.2 million base64 characters: more than a whole line may hold
MESSAGES: List[Any] = [{"role": "user", "content": "Say it"}]


def omitted(payload: str) -> Dict[str, Any]:
    return {"type": "media_omitted", "bytes": len(payload)}


def test_audio_in_a_chat_response_is_omitted_and_the_caller_still_gets_it(rig: Any) -> None:
    audio = {"id": "audio_1", "data": BIG, "expires_at": 1, "transcript": "Hello"}
    rig.server.on(CHAT_PATH, Reply(body=chat_body(None, audio=audio)))
    with rig.client() as client:
        result = client.chat.completions.create(model=MODEL, messages=MESSAGES)
    returned = result.choices[0].message.audio
    assert returned is not None
    assert returned.data == BIG
    (line,) = rig.lines()
    recorded = line["response"]["choices"][0]["message"]["audio"]
    assert recorded == {
        "id": "audio_1",
        "data": omitted(BIG),
        "expires_at": 1,
        "transcript": "Hello",
    }
    assert len(rig.raw()[0]) < 5_000


def test_input_audio_and_images_in_a_request_make_the_call_multimodal(rig: Any) -> None:
    rig.server.on(CHAT_PATH, Reply(body=chat_body()))
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "input_audio", "input_audio": {"data": BIG, "format": "wav"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + BIG}},
    ]
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": content}])
    (line,) = rig.lines()
    parts = line["request"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "What is this?"}
    assert [part["type"] for part in parts[1:]] == ["media_omitted", "media_omitted"]
    assert line["norm"]["has_media"] is True


def test_images_returned_in_a_response_list_are_omitted(rig: Any) -> None:
    images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + BIG}}]
    rig.server.on(CHAT_PATH, Reply(body=chat_body("Here", images=images)))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    (line,) = rig.lines()
    (image,) = line["response"]["choices"][0]["message"]["images"]
    assert image["type"] == "media_omitted"
    assert line["norm"]["response"] == "Here"


def test_streamed_audio_is_assembled_as_a_transcript_and_a_size(rig: Any) -> None:
    def chunk(delta: Dict[str, Any]) -> Event:
        choice = {"index": 0, "delta": delta, "finish_reason": None}
        body = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": MODEL}
        return None, dict(body, choices=[choice])

    events: List[Event] = [
        chunk(
            {"role": "assistant", "audio": {"id": "audio_9", "data": "AAAA", "transcript": "Hel"}}
        ),
        chunk({"audio": {"data": "BBBBBB", "transcript": "lo"}}),
        chunk({"audio": {"expires_at": 5}}),
        (
            None,
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ),
        (None, "[DONE]"),
    ]
    rig.server.on(CHAT_PATH, Reply(events=events))
    with rig.client() as client:
        list(client.chat.completions.create(model=MODEL, messages=MESSAGES, stream=True))
    (line,) = rig.lines()
    audio = line["response"]["choices"][0]["message"]["audio"]
    assert audio == {
        "id": "audio_9",
        "transcript": "Hello",
        "data": {"type": "media_omitted", "bytes": 10},
    }


def image_item(result: str) -> Dict[str, Any]:
    return {"id": "ig_1", "type": "image_generation_call", "status": "completed", "result": result}


def test_a_generated_image_in_a_responses_output_is_omitted(rig: Any) -> None:
    body = response_body()
    body["output"] = [image_item(BIG), *body["output"]]
    rig.server.on(RESPONSES_PATH, Reply(body=body))
    with rig.client() as client:
        client.responses.create(model=MODEL, input="Draw")
    (line,) = rig.lines()
    assert line["response"]["output"][0]["result"] == omitted(BIG)
    assert line["norm"]["response"] == "Hello there"


def test_a_generated_image_sent_back_in_a_responses_request_is_omitted(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(body=response_body()))
    previous = [image_item(BIG), {"role": "user", "content": "Again"}]
    with rig.client() as client:
        client.responses.create(model=MODEL, input=previous)
    (line,) = rig.lines()
    assert line["request"]["input"][0]["result"] == omitted(BIG)
    assert line["request"]["input"][1] == {"role": "user", "content": "Again"}


def test_audio_parts_and_a_generated_image_in_a_responses_stream_are_omitted(rig: Any) -> None:
    final = response_events()
    done = final[-1][1]["response"]
    done["output"][0]["content"].append({"type": "output_audio", "data": BIG, "transcript": "Hi"})
    done["output"].append(image_item(BIG))
    rig.server.on(RESPONSES_PATH, Reply(events=final))
    with rig.client() as client:
        list(client.responses.create(model=MODEL, input="Hi", stream=True))
    (line,) = rig.lines()
    output = line["response"]["output"]
    assert output[0]["content"][1] == {
        "type": "output_audio",
        "data": omitted(BIG),
        "transcript": "Hi",
    }
    assert output[-1]["result"] == omitted(BIG)


@pytest.mark.parametrize(
    "choices", [None, ["not a choice"], [{"index": 0, "finish_reason": "stop"}]]
)
def test_odd_chat_response_shapes_are_recorded_without_media_handling(
    rig: Any, choices: Any
) -> None:
    rig.server.on(CHAT_PATH, Reply(body=dict(chat_body(), choices=choices)))
    with rig.client() as client:
        client.chat.completions.create(model=MODEL, messages=MESSAGES)
    assert len(rig.lines()) == 1


def test_odd_responses_output_items_are_recorded_without_media_handling(rig: Any) -> None:
    rig.server.on(RESPONSES_PATH, Reply(body=dict(response_body(), output=[1, {"content": "x"}])))
    with rig.client() as client:
        client.responses.create(model=MODEL, input="Hi")
    assert len(rig.lines()) == 1
