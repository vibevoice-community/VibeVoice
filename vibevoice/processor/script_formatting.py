import re

SPEAKER_LINE_PATTERN = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def format_conversation_script(script: str, num_speakers: int) -> str:
    """Normalize UI script text while preserving explicit speaker turns."""
    if num_speakers < 1:
        raise ValueError("Number of speakers must be at least 1.")

    turns: list[tuple[int, list[str]]] = []
    current_speaker = 1
    current_parts: list[str] = []

    def flush_turn() -> None:
        nonlocal current_parts
        if current_parts:
            turns.append((current_speaker, current_parts))
            current_parts = []

    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        speaker_match = SPEAKER_LINE_PATTERN.match(line)
        if speaker_match:
            speaker_id = int(speaker_match.group(1))
            if speaker_id < 1 or speaker_id > num_speakers:
                raise ValueError(
                    f"Speaker {speaker_id} is outside the selected range of "
                    f"Speaker 1 through Speaker {num_speakers}."
                )

            flush_turn()
            current_speaker = speaker_id
            speaker_text = speaker_match.group(2).strip()
            if speaker_text:
                current_parts.append(speaker_text)
            continue

        current_parts.append(line)

    flush_turn()

    if not turns:
        raise ValueError("Please provide at least one non-empty script line.")

    return "\n".join(
        f"Speaker {speaker_id}: {' '.join(parts)}"
        for speaker_id, parts in turns
    )
