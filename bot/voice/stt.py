"""
Azure Speech-to-Text (STT) service.

Provides speech-to-text conversion using Azure Speech Services.
Can be tested independently without running the Discord bot.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

try:
    import azure.cognitiveservices.speech as speech
    _AZURE_SPEECH_AVAILABLE = True
except ImportError:
    _AZURE_SPEECH_AVAILABLE = False
    logging.warning("azure-cognitiveservices-speech not installed - STT disabled")

from .config import VoiceConfig


logger = logging.getLogger(__name__)


class AzureSTT:
    """
    Azure Speech-to-Text service.
    
    Provides speech-to-text conversion using Azure Speech Services.
    Can be tested independently without running the Discord bot.
    """
    
    def __init__(self, config: VoiceConfig):
        """
        Initialize Azure STT with the given config.
        
        Args:
            config: VoiceConfig with Azure Speech credentials
            
        Raises:
            ImportError: If azure-cognitiveservices-speech is not installed
            ValueError: If config is not properly configured
        """
        if not _AZURE_SPEECH_AVAILABLE:
            raise ImportError("azure-cognitiveservices-speech not installed")
        
        if not config.is_configured:
            raise ValueError("Azure Speech not configured (missing key or region)")
        
        self.config = config
        self._speech_config = speech.SpeechConfig(
            subscription=config.key,
            region=config.region
        )
        
        # Set output format for better recognition
        self._speech_config.output_format = speech.OutputFormat.Detailed
        
        if config.endpoint:
            self._speech_config.endpoint = config.endpoint
    
    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Audio data to transcribe (supports WAV, OGG, MP3)
            language: Language code (e.g., "en-US"). Auto-detects if not specified.
            
        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If transcription fails
        """
        logger.info(f"STT: Starting transcription, audio size={len(audio_bytes)} bytes, language={language}")
        
        # Create a FRESH speech config for each transcription to avoid language state issues
        speech_config = speech.SpeechConfig(
            subscription=self.config.key,
            region=self.config.region
        )
        speech_config.output_format = speech.OutputFormat.Detailed
        
        if self.config.endpoint:
            speech_config.endpoint = self.config.endpoint
        
        # Set language on the fresh config BEFORE creating recognizer
        if language:
            speech_config.speech_recognition_language = language
            logger.info(f"STT: Language set to {language}")
        
        # Create audio input from bytes - specify format for 16kHz mono WAV
        audio_stream = speech.audio.PushAudioInputStream(
            speech.audio.AudioStreamFormat(
                samples_per_second=16000,
                channels=1,
                bits_per_sample=16
            )
        )
        
        # Write audio data to stream
        audio_stream.write(audio_bytes)
        audio_stream.close()
        
        audio_config = speech.audio.AudioConfig(stream=audio_stream)
        
        # Create recognizer with the fresh config
        recognizer = speech.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        # Also set language on recognizer to be sure
        if language:
            recognizer.speech_recognition_language = language
        
        # Synchronous recognition
        result = recognizer.recognize_once()
        
        logger.info(f"STT: Result reason={result.reason}, text='{result.text if result.text else ''}'")
        
        if result.reason == speech.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speech.ResultReason.NoMatch:
            no_match_details = result.no_match_details
            logger.warning(f"No speech recognized in audio. NoMatch details: {no_match_details}")
            return ""
        elif result.reason == speech.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"STT cancelled: {cancellation.reason}"
            if cancellation.error_details:
                error_msg += f" - {cancellation.error_details}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            raise RuntimeError(f"STT failed with reason: {result.reason}")
    
    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            file_path: Path to audio file (supports WAV, OGG, MP3)
            language: Language code (e.g., "en-US"). Auto-detects if not specified.
            
        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If transcription fails or file cannot be read
        """
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
        except IOError as e:
            raise RuntimeError(f"Failed to read audio file: {e}")
        
        return self.transcribe(audio_bytes, language)
    
    def transcribe_from_url(self, url: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio from a URL.
        
        Args:
            url: URL to audio file
            language: Language code (e.g., "en-US"). Auto-detects if not specified.
            
        Returns:
            Transcribed text
            
        Raises:
            RuntimeError: If transcription fails
        """
        import urllib.request
        
        try:
            with urllib.request.urlopen(url) as response:
                audio_bytes = response.read()
        except Exception as e:
            raise RuntimeError(f"Failed to download audio from URL: {e}")
        
        return self.transcribe(audio_bytes, language)


def create_stt(config: Optional[VoiceConfig]) -> Optional[AzureSTT]:
    """
    Factory function to create AzureSTT instance.
    
    Args:
        config: VoiceConfig with Azure Speech credentials
        
    Returns:
        AzureSTT instance if config is valid, None otherwise
    """
    if not config or not config.is_configured:
        return None
    
    try:
        return AzureSTT(config)
    except Exception as e:
        logger.error(f"Failed to create AzureSTT: {e}")
        return None