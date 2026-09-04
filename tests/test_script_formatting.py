import unittest

from vibevoice.processor.script_formatting import format_conversation_script
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


class FormatConversationScriptTests(unittest.TestCase):
    def test_blank_lines_and_continuations_do_not_change_speaker(self) -> None:
        script = """Speaker 1: First paragraph.

Second paragraph.
Speaker 2: Reply.

More from speaker two.
Speaker 1: Closing."""

        result = format_conversation_script(script, num_speakers=2)

        self.assertEqual(
            result,
            "Speaker 1: First paragraph. Second paragraph.\n"
            "Speaker 2: Reply. More from speaker two.\n"
            "Speaker 1: Closing.",
        )

    def test_unlabeled_script_defaults_to_speaker_one(self) -> None:
        script = """Opening paragraph.

Still the same speaker."""

        result = format_conversation_script(script, num_speakers=2)

        self.assertEqual(
            result,
            "Speaker 1: Opening paragraph. Still the same speaker.",
        )

    def test_speaker_labels_are_case_insensitive(self) -> None:
        result = format_conversation_script(
            "speaker 1: Hello.\nSPEAKER 2: Hi.",
            num_speakers=2,
        )

        self.assertEqual(result, "Speaker 1: Hello.\nSpeaker 2: Hi.")

    def test_one_based_ui_labels_map_to_zero_based_voice_prompts(self) -> None:
        formatted_script = format_conversation_script(
            "Speaker 1: First.\nSpeaker 2: Second.",
            num_speakers=2,
        )

        parsed_script = VibeVoiceProcessor._parse_script(None, formatted_script)

        self.assertEqual(
            [speaker_id for speaker_id, _ in parsed_script],
            [0, 1],
        )

    def test_speaker_outside_selected_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the selected range"):
            format_conversation_script("Speaker 3: Hello.", num_speakers=2)


if __name__ == "__main__":
    unittest.main()
