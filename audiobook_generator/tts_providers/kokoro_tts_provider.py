import html
import io
import logging
import math
import os
from datetime import datetime, timedelta
from time import sleep
import requests

from kokoro_onnx import Kokoro
import soundfile as sf

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.utils import split_text, set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

MAX_RETRIES = 12  # Max_retries constant for network errors



class KokoroTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        # TTS provider specific config
        config.voice_name = config.voice_name or "am_michael"
        config.output_format = config.output_format or "wav"
        self.kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
        super().__init__(config)

    def __str__(self) -> str:
        return (
                super().__str__()
                + f", voice_name={self.config.voice_name}, language={self.config.language}, break_duration={self.config.break_duration}, output_format={self.config.output_format}"
        )

    def is_access_token_expired(self) -> bool:
        return False

    def auto_renew_access_token(self) -> str:
        return ""

    def get_access_token(self) -> str:
        return ""

    def text_to_speech(
            self,
            text: str,
            output_file: str,
            audio_tags: AudioTags,
    ):
        # Adjust this value based on your testing
        max_chars = 1800 if self.config.language.startswith("zh") else 3000

        text_chunks = split_text(text, max_chars, self.config.language)

        audio_segments = []
        sample_rate = None  # 初始化 sample_rate

        for i, chunk in enumerate(text_chunks, 1):
            logger.debug(
                f"Processing chunk {i} of {len(text_chunks)}, length={len(chunk)}, text=[{chunk}]"
            )
            escaped_text = html.escape(chunk)
            logger.debug(f"Escaped text: [{escaped_text}]")
            # replace MAGIC_BREAK_STRING with a break tag for section/paragraph break
            escaped_text = escaped_text.replace(
                self.get_break_string().strip(),
                f" <break time='{self.config.break_duration}ms' /> ",
            )  # strip in case leading bank is missing
            logger.info(
                f"Processing chapter-{audio_tags.idx} <{audio_tags.title}>, chunk {i} of {len(text_chunks)}"
            )
            ssml = f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{self.config.language}'><voice name='{self.config.voice_name}'>{escaped_text}</voice></speak>"
            logger.debug(f"SSML: [{ssml}]")

            for retry in range(MAX_RETRIES):
                try:
                    logger.info(
                        "Sending request to kokoro TTS, data length: " + str(len(ssml))
                    )
                    if not self.config.language:
                        logger.warning("Language not specified, using default 'en-US'.")
                        self.config.language = "en-us"
                    else:
                        ## 转小写
                        self.config.language = self.config.language.lower()

                    samples, sample_rate = self.kokoro.create(
                            chunk, voice=self.config.voice_name, speed=1.0, lang=self.config.language
                        )
                    # 将生成的音频数据添加到列表中
                    audio_segments.extend(samples)
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning(
                        f"Error while converting text to speech (attempt {retry + 1}): {e}"
                    )
                    if retry < MAX_RETRIES - 1:
                        sleep(2 ** retry)
                    else:
                        raise e
            # 合并音频并保存
            if audio_segments:
                sf.write(output_file, audio_segments, sample_rate)

        set_audio_tags(output_file, audio_tags)

    def get_break_string(self):
        return " "

    def get_output_file_extension(self):
        if self.config.output_format.startswith("amr"):
            return "amr"
        elif self.config.output_format.startswith("ogg"):
            return "ogg"
        elif self.config.output_format.endswith("truesilk"):
            return "silk"
        elif self.config.output_format.endswith("pcm"):
            return "pcm"
        elif self.config.output_format.startswith("raw"):
            return "wav"
        elif self.config.output_format.startswith("webm"):
            return "webm"
        elif self.config.output_format.endswith("opus"):
            return "opus"
        elif self.config.output_format.endswith("mp3"):
            return "mp3"
        else:
           return "wav"

    def validate_config(self):
        # TODO: Need to dig into Azure properties, im not familiar with them, but look at OpenAI as ref example
        pass

    def estimate_cost(self, total_chars):
        return 0
