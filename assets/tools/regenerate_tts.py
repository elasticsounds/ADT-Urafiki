#!/usr/bin/env python3
"""
Unified TTS Regeneration Script for ADT Press

A self-contained script that regenerates text-to-speech (TTS) audio files for an
existing ADT output. Supports both Azure and OpenAI TTS providers with no external
TTS library dependencies.

Requirements:
    pip install requests pydub

Environment Variables:
    For Azure TTS:
        AZURE_SPEECH_KEY - Your Azure Speech API key
        AZURE_SPEECH_REGION - Your Azure region (e.g., "eastus")

    For OpenAI TTS:
        OPENAI_API_KEY - Your OpenAI API key

Usage:
    # Regenerate all audio for all languages
    python regenerate_tts.py

    # Regenerate from a specific directory
    python regenerate_tts.py --adt-dir output/urafiki-combine

    # Regenerate specific language
    python regenerate_tts.py --languages en-tz

    # Regenerate specific text IDs
    python regenerate_tts.py --text-ids txt_p1_g0_t0,txt_p1_g0_t0_easy_read

    # Force a specific provider
    python regenerate_tts.py --provider openai

    # Dry run to see what would be regenerated
    python regenerate_tts.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

# Only required dependencies
import requests

# Optional: pydub for audio transcoding (graceful fallback if not available)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Audio transcoding will be skipped.")
    print("Install with: pip install pydub")


# ============================================================================
# Embedded Voice Configurations
# ============================================================================

AZURE_VOICES = {
    "default": "en-US-JennyNeural",
    # English variants
    "en": "en-US-JennyNeural",
    "en-us": "en-US-JennyNeural",
    "en-gb": "en-GB-SoniaNeural",
    "en-au": "en-AU-NatashaNeural",
    "en-ca": "en-CA-ClaraNeural",
    "en-in": "en-IN-NeerjaNeural",
    "en-ie": "en-IE-EmilyNeural",
    "en-nz": "en-NZ-MollyNeural",
    "en-za": "en-ZA-LeahNeural",
    "en-tz": "en-TZ-ElimuNeural",
    # Spanish variants
    "es": "es-ES-ElviraNeural",
    "es-es": "es-ES-ElviraNeural",
    "es-mx": "es-MX-DaliaNeural",
    "es-ar": "es-AR-ElenaNeural",
    "es-co": "es-CO-SalomeNeural",
    "es-uy": "es-UY-ValentinaNeural",
    # Portuguese variants
    "pt": "pt-BR-FranciscaNeural",
    "pt-br": "pt-BR-FranciscaNeural",
    "pt-pt": "pt-PT-RaquelNeural",
    # French variants
    "fr": "fr-FR-DeniseNeural",
    "fr-ca": "fr-CA-SylvieNeural",
    # South Asian languages
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "ta-in": "ta-IN-PallaviNeural",
    "ta-lk": "ta-LK-SaranyaNeural",
    "si": "si-LK-ThiliniNeural",
    # African languages
    "sw": "sw-KE-ZuriNeural",
    "sw-ke": "sw-KE-ZuriNeural",
    "sw-tz": "sw-TZ-DaudiNeural",
    "af": "af-ZA-AdriNeural",
    "am": "am-ET-MekdesNeural",
    "zu": "zu-ZA-ThandoNeural",
    # Arabic
    "ar": "ar-SA-ZariyahNeural",
    "ar-eg": "ar-EG-SalmaNeural",
    # Other common languages
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "tr": "tr-TR-EmelNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "th": "th-TH-PremwadeeNeural",
    "id": "id-ID-GadisNeural",
    "ms": "ms-MY-YasminNeural",
    "fil": "fil-PH-BlessicaNeural",
}

OPENAI_VOICES = {
    "default": "alloy",
    "en": "alloy",
}

VOICE_MAPS = {
    "azure": AZURE_VOICES,
    "openai": OPENAI_VOICES,
}

# Speech instructions for accent/pronunciation (used by OpenAI TTS)
SPEECH_INSTRUCTIONS = {
    "default": "Speak in a cheerful and positive tone.",
    "en": "Speak in a cheerful and positive tone.",
    "en-tz": """Speak it in a Tanzanian English accent. Use the pronunciation and
intonation typical of Tanzania. Maintain a calm, conversational tone.
Speak in a cheerful and positive tone.""",
    "sw-tz": """Speak it in Tanzanian Swahili (Kiswahili cha Tanzania). Use the
pronunciation and intonation typical of mainland Tanzania, not Kenyan Swahili.
Speak in a cheerful and positive tone.""",
    "pt": """Speak it in a European Portuguese (Portugal) accent. Use the pronunciation
and intonation typical of Portugal, not Brazilian Portuguese.
Speak in a cheerful and positive tone.""",
    "pt-br": """Speak it in a Brazilian Portuguese accent. Use the pronunciation and
intonation typical of Brazil. Speak in a cheerful and positive tone.""",
    "es-uy": """Speak it in an Uruguayan Rioplatense accent. Pronounce the 'y' and 'll'
sounds like 'sh' in 'shoe'. Speak in a cheerful and positive tone.""",
}


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class TTSConfig:
    """Complete TTS configuration."""

    provider: str = "azure"  # "azure" or "openai"
    format: str = "mp3"
    bit_rate: str = "64k"
    sample_rate: int = 24000

    # Azure settings
    azure_key: str = ""
    azure_region: str = ""

    # OpenAI settings
    openai_key: str = ""
    openai_model: str = "tts-1"

    def __post_init__(self):
        # Load from environment if not set
        if not self.azure_key:
            self.azure_key = os.environ.get("AZURE_SPEECH_KEY", "")
        if not self.azure_region:
            self.azure_region = os.environ.get("AZURE_SPEECH_REGION", "")
        if not self.openai_key:
            self.openai_key = os.environ.get("OPENAI_API_KEY", "")


@dataclass
class SpeechFile:
    """Metadata for a generated speech file."""

    speech_id: str
    text_id: str
    speech_path: str
    language_code: str
    provider: str
    voice: str


# ============================================================================
# Voice and Instruction Resolution
# ============================================================================


def resolve_voice(provider: str, language_code: str) -> str:
    """Resolve the appropriate voice based on provider and language."""
    voice_map = VOICE_MAPS.get(provider, AZURE_VOICES)
    normalized = language_code.lower()

    # Try exact match first (e.g., "en-tz")
    if normalized in voice_map:
        return voice_map[normalized]

    # Try base language (e.g., "en" from "en-tz")
    base_lang = normalized.split("-")[0]
    if base_lang in voice_map:
        return voice_map[base_lang]

    return voice_map.get("default", "en-US-JennyNeural")


def resolve_instructions(language_code: str) -> str:
    """Resolve accent/pronunciation instructions for a language."""
    normalized = language_code.lower()

    if normalized in SPEECH_INSTRUCTIONS:
        return SPEECH_INSTRUCTIONS[normalized]

    base_lang = normalized.split("-")[0]
    if base_lang in SPEECH_INSTRUCTIONS:
        return SPEECH_INSTRUCTIONS[base_lang]

    return SPEECH_INSTRUCTIONS.get("default", "")


def get_language_name(language_code: str) -> str:
    """Get human-readable language name from code."""
    # Simple mapping for common languages
    LANGUAGE_NAMES = {
        "en": "English", "en-us": "English (US)", "en-gb": "English (UK)",
        "en-tz": "English (Tanzania)", "sw": "Swahili", "sw-tz": "Swahili (Tanzania)",
        "sw-ke": "Swahili (Kenya)", "es": "Spanish", "es-uy": "Spanish (Uruguay)",
        "pt": "Portuguese", "pt-br": "Portuguese (Brazil)", "fr": "French",
        "de": "German", "it": "Italian", "ar": "Arabic", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "ta": "Tamil",
        "si": "Sinhala", "ru": "Russian",
    }
    return LANGUAGE_NAMES.get(language_code.lower(), language_code)


# ============================================================================
# Text Utilities
# ============================================================================


EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+"
)


def strip_emojis(text: str) -> str:
    """Remove all emoji codepoints from the provided string."""
    if not text:
        return text
    return EMOJI_PATTERN.sub("", text)


def is_speakable_text(text: str) -> bool:
    """Check if text contains speakable content suitable for TTS."""
    if not text or not text.strip():
        return False
    stripped = text.strip()
    return any(unicodedata.category(c).startswith(("L", "N")) for c in stripped)


# ============================================================================
# TTS API Implementations
# ============================================================================


def generate_azure_tts(
    text: str,
    voice: str,
    output_path: str,
    api_key: str,
    region: str,
    output_format: str = "audio-24khz-48kbitrate-mono-mp3",
) -> bool:
    """
    Generate speech using Azure Cognitive Services TTS.

    Args:
        text: Text to convert to speech
        voice: Azure voice name (e.g., "en-TZ-ElimuNeural")
        output_path: Path to save the audio file
        api_key: Azure Speech API key
        region: Azure region (e.g., "eastus")
        output_format: Azure output format

    Returns:
        True if successful, False otherwise
    """
    # Extract language from voice name (e.g., "en-TZ" from "en-TZ-ElimuNeural")
    voice_parts = voice.split("-")
    lang = f"{voice_parts[0]}-{voice_parts[1]}" if len(voice_parts) >= 2 else "en-US"

    # Build SSML
    ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{lang}'>
    <voice name='{voice}'>{text}</voice>
</speak>"""

    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": output_format,
        "User-Agent": "ADT-Press-TTS",
    }

    try:
        response = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return True
    except requests.exceptions.RequestException as e:
        print(f"    Azure TTS error: {e}")
        return False


def generate_openai_tts(
    text: str,
    voice: str,
    output_path: str,
    api_key: str,
    model: str = "tts-1",
    instructions: str = "",
) -> bool:
    """
    Generate speech using OpenAI TTS API.

    Args:
        text: Text to convert to speech
        voice: OpenAI voice name (e.g., "alloy", "echo", "fable", "onyx", "nova", "shimmer")
        output_path: Path to save the audio file
        api_key: OpenAI API key
        model: OpenAI TTS model (tts-1 or tts-1-hd)
        instructions: Optional instructions for pronunciation/accent

    Returns:
        True if successful, False otherwise
    """
    url = "https://api.openai.com/v1/audio/speech"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
    }

    # Add instructions if provided (for models that support it)
    if instructions and "tts" in model:
        payload["instructions"] = instructions

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return True
    except requests.exceptions.RequestException as e:
        print(f"    OpenAI TTS error: {e}")
        return False


# ============================================================================
# Main TTS Generation
# ============================================================================


def generate_speech_file(
    output_dir: str,
    config: TTSConfig,
    language_code: str,
    text_id: str,
    text: str,
) -> SpeechFile | None:
    """
    Generate a speech file from text using TTS.

    Args:
        output_dir: Output directory for audio files
        config: TTS configuration
        language_code: Target language for speech
        text_id: Unique identifier for the text
        text: Text content to convert to speech

    Returns:
        SpeechFile metadata object or None if failed
    """
    # Sanitize text
    sanitized_text = strip_emojis(text)
    if not sanitized_text.strip():
        sanitized_text = text

    if not sanitized_text or not sanitized_text.strip():
        return None

    # Setup paths - output directly to audio folder with simple filename
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    output_filename = f"{text_id}.{config.format}"
    output_path = os.path.join(audio_dir, output_filename)

    # Resolve voice
    resolved_voice = resolve_voice(config.provider, language_code)

    # Handle non-speakable text
    if not is_speakable_text(sanitized_text):
        # Create a short silent audio file
        if PYDUB_AVAILABLE:
            silence = AudioSegment.silent(duration=100)
            silence.export(output_path, format=config.format, bitrate=config.bit_rate)
        else:
            # Create minimal valid MP3 (silent frame)
            # This is a minimal valid MP3 file with silence
            with open(output_path, "wb") as f:
                f.write(b'\xff\xfb\x90\x00' + b'\x00' * 417)

        return SpeechFile(
            speech_id=f"{text_id}_{language_code}",
            text_id=text_id,
            speech_path=output_filename,
            language_code=language_code,
            provider=config.provider,
            voice=resolved_voice,
        )

    # Generate TTS based on provider
    success = False

    if config.provider == "azure":
        if not config.azure_key or not config.azure_region:
            print("    Error: Azure credentials not configured")
            print("    Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables")
            return None

        success = generate_azure_tts(
            text=sanitized_text,
            voice=resolved_voice,
            output_path=output_path,
            api_key=config.azure_key,
            region=config.azure_region,
        )

    elif config.provider == "openai":
        if not config.openai_key:
            print("    Error: OpenAI API key not configured")
            print("    Set OPENAI_API_KEY environment variable")
            return None

        instructions = resolve_instructions(language_code)
        language_name = get_language_name(language_code)
        full_instructions = f"The following text is in {language_name}.\n\n{instructions}"

        success = generate_openai_tts(
            text=sanitized_text,
            voice=resolved_voice,
            output_path=output_path,
            api_key=config.openai_key,
            model=config.openai_model,
            instructions=full_instructions,
        )

    if not success:
        return None

    # Optionally transcode with pydub
    if PYDUB_AVAILABLE and os.path.exists(output_path):
        try:
            audio = AudioSegment.from_file(output_path, format=config.format)
            audio = audio.set_frame_rate(config.sample_rate)
            audio.export(
                output_path,
                format=config.format,
                bitrate=config.bit_rate,
                parameters=["-ac", "1"],
            )
        except Exception as e:
            print(f"    Warning: Could not transcode {text_id}: {e}")

    return SpeechFile(
        speech_id=f"{text_id}_{language_code}",
        text_id=text_id,
        speech_path=output_filename,
        language_code=language_code,
        provider=config.provider,
        voice=resolved_voice,
    )


# ============================================================================
# File Utilities
# ============================================================================


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    """Save a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def discover_languages(adt_dir: str) -> list[str]:
    """Discover available languages in the ADT output."""
    i18n_dir = os.path.join(adt_dir, "adt", "content", "i18n")
    if not os.path.exists(i18n_dir):
        return []
    return [d for d in os.listdir(i18n_dir) if os.path.isdir(os.path.join(i18n_dir, d))]


def load_texts_for_language(adt_dir: str, language: str) -> dict[str, str]:
    """Load text translations for a language."""
    texts_path = os.path.join(adt_dir, "adt", "content", "i18n", language, "texts.json")
    if os.path.exists(texts_path):
        return load_json(texts_path)
    return {}


def load_audios_for_language(adt_dir: str, language: str) -> dict[str, str]:
    """Load existing audio mappings for a language."""
    audios_path = os.path.join(adt_dir, "adt", "content", "i18n", language, "audios.json")
    if os.path.exists(audios_path):
        return load_json(audios_path)
    return {}


# ============================================================================
# Main Regeneration Logic
# ============================================================================


def regenerate_tts(
    adt_dir: str,
    config: TTSConfig,
    languages: list[str] | None = None,
    text_ids: list[str] | None = None,
    easy_read_only: bool = False,
    dry_run: bool = False,
    delay: float = 0.5,
) -> dict[str, dict[str, SpeechFile]]:
    """
    Regenerate TTS audio files for an ADT output.

    Args:
        adt_dir: Path to ADT output directory
        config: TTS configuration
        languages: Languages to regenerate (None = all)
        text_ids: Specific text IDs to regenerate (None = all)
        easy_read_only: Only regenerate easy_read variants
        dry_run: Print what would be done without doing it

    Returns:
        Dictionary of language -> text_id -> SpeechFile
    """
    # Discover available languages
    available_languages = discover_languages(adt_dir)
    if not available_languages:
        print(f"No languages found in {adt_dir}/adt/content/i18n/")
        return {}

    # Filter languages if specified
    if languages:
        target_languages = [lang for lang in languages if lang in available_languages]
        if not target_languages:
            print(f"None of the specified languages found. Available: {available_languages}")
            return {}
    else:
        target_languages = available_languages

    print(f"Target languages: {target_languages}")
    print(f"Provider: {config.provider}")
    print(f"Format: {config.format}, Bitrate: {config.bit_rate}, Sample rate: {config.sample_rate}")
    print()

    results: dict[str, dict[str, SpeechFile]] = {}

    for language in target_languages:
        print(f"Processing language: {language}")

        # Resolve voice for this language
        voice = resolve_voice(config.provider, language)
        print(f"  Provider: {config.provider}, Voice: {voice}")

        # Load texts
        texts = load_texts_for_language(adt_dir, language)
        if not texts:
            print(f"  No texts found for {language}")
            continue

        # Filter text IDs
        target_text_ids = list(texts.keys())
        if text_ids:
            target_text_ids = [t for t in target_text_ids if t in text_ids]
        if easy_read_only:
            target_text_ids = [t for t in target_text_ids if t.endswith("_easy_read")]

        print(f"  Found {len(target_text_ids)} texts to regenerate")

        if dry_run:
            for tid in target_text_ids[:5]:
                text_preview = texts[tid][:50] + "..." if len(texts[tid]) > 50 else texts[tid]
                print(f'    Would regenerate: {tid} -> "{text_preview}"')
            if len(target_text_ids) > 5:
                print(f"    ... and {len(target_text_ids) - 5} more")
            continue

        # Generate TTS for each text
        language_dir = os.path.join(adt_dir, "adt", "content", "i18n", language)
        language_results: dict[str, SpeechFile] = {}
        audios_map: dict[str, str] = load_audios_for_language(adt_dir, language)

        for i, tid in enumerate(target_text_ids):
            text = texts[tid]
            result = generate_speech_file(
                output_dir=language_dir,
                config=config,
                language_code=language,
                text_id=tid,
                text=text,
            )

            if result:
                language_results[tid] = result
                audios_map[tid] = result.speech_path
                print(f"    [{i+1}/{len(target_text_ids)}] Generated: {tid}")
            else:
                print(f"    [{i+1}/{len(target_text_ids)}] Failed: {tid}")

            if delay > 0 and i < len(target_text_ids) - 1:
                time.sleep(delay)

        results[language] = language_results

        # Save updated audios.json
        audios_path = os.path.join(language_dir, "audios.json")
        save_json(audios_path, audios_map)
        print(f"  Updated {audios_path}")
        print(f"  Completed {len(language_results)} files for {language}")
        print()

    return results


# ============================================================================
# CLI
# ============================================================================


def find_adt_dir() -> str | None:
    """Try to find an ADT output directory in the current working directory."""
    cwd = os.getcwd()

    # Check if current dir is an ADT output
    if os.path.exists(os.path.join(cwd, "adt", "content", "i18n")):
        return cwd

    # Check output/ subdirectory
    output_dir = os.path.join(cwd, "output")
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "adt", "content", "i18n")):
                    return item_path

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate TTS audio files for an ADT output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect ADT directory and regenerate all
  python regenerate_tts.py

  # Specify ADT directory
  python regenerate_tts.py --adt-dir output/urafiki-combine

  # Regenerate specific language
  python regenerate_tts.py --languages en-tz,sw-tz

  # Force OpenAI provider
  python regenerate_tts.py --provider openai

  # Dry run
  python regenerate_tts.py --dry-run

Environment Variables:
  AZURE_SPEECH_KEY      Azure Speech API key
  AZURE_SPEECH_REGION   Azure region (e.g., eastus)
  OPENAI_API_KEY        OpenAI API key
        """,
    )

    parser.add_argument("--adt-dir", "-d", help="Path to ADT output directory")
    parser.add_argument("--languages", "-l", help="Comma-separated list of languages to regenerate")
    parser.add_argument("--text-ids", "-t", help="Comma-separated list of text IDs to regenerate")
    parser.add_argument("--easy-read-only", "-e", action="store_true", help="Only regenerate easy_read variants")
    parser.add_argument("--provider", "-p", choices=["azure", "openai"], default="azure", help="TTS provider")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print what would be done without doing it")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between TTS requests (default: 0.5)")

    args = parser.parse_args()

    # Find ADT directory
    adt_dir = args.adt_dir
    if not adt_dir:
        adt_dir = find_adt_dir()
        if not adt_dir:
            print("Error: Could not find ADT output directory.")
            print("Run from an ADT project folder or specify --adt-dir")
            sys.exit(1)

    # Validate ADT directory
    if not os.path.exists(adt_dir):
        print(f"Error: ADT directory not found: {adt_dir}")
        sys.exit(1)

    adt_content_dir = os.path.join(adt_dir, "adt", "content", "i18n")
    if not os.path.exists(adt_content_dir):
        print(f"Error: Invalid ADT directory structure. Expected {adt_content_dir}")
        sys.exit(1)

    # Create configuration
    config = TTSConfig(provider=args.provider)

    # Parse arguments
    languages = args.languages.split(",") if args.languages else None
    text_ids = args.text_ids.split(",") if args.text_ids else None

    print("=" * 60)
    print("ADT Press TTS Regeneration")
    print("=" * 60)
    print(f"ADT Directory: {adt_dir}")
    print(f"Provider: {args.provider}")
    if languages:
        print(f"Languages: {languages}")
    if text_ids:
        print(f"Text IDs: {text_ids}")
    if args.easy_read_only:
        print("Mode: Easy Read only")
    if args.dry_run:
        print("Mode: DRY RUN")
    print("=" * 60)
    print()

    # Run regeneration
    results = regenerate_tts(
        adt_dir=adt_dir,
        config=config,
        languages=languages,
        text_ids=text_ids,
        easy_read_only=args.easy_read_only,
        dry_run=args.dry_run,
        delay=args.delay,
    )

    # Summary
    if not args.dry_run:
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        total = 0
        for lang, files in results.items():
            print(f"  {lang}: {len(files)} files regenerated")
            total += len(files)
        print(f"  Total: {total} files")


if __name__ == "__main__":
    main()
