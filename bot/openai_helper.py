from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import httpx
import openai
import tiktoken
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from plugin_manager import PluginManager
from utils import decode_image, encode_image, is_direct_result

# Models can be found here: https://platform.openai.com/docs/models/overview
# Updated with 2025 models including GPT-5 and GPT-4.1
GPT_3_MODELS = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-instruct",
}
GPT_3_16K_MODELS = {
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
}
GPT_4_MODELS = {
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo-preview",
}
GPT_4_32K_MODELS = {
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
}
GPT_4_128K_MODELS = {
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
}
GPT_4O_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "chatgpt-4o-latest",
    "gpt-4o-mini-realtime-preview",
}
GPT_4_1_MODELS = {"gpt-4.1", "gpt-4.1-mini"}
GPT_5_MODELS = {
    "gpt-5",
    "gpt-5-2025-08-07",
    "gpt-5-mini",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano",
    "gpt-5-nano-2025-08-07",
    "gpt-5-chat",
    "gpt-5-chat-latest",
}
O_MODELS = {"o1", "o1-mini", "o1-preview", "o3-mini", "o4-mini"}
GPT_VISION_MODELS = {
    "gpt-4o",
    "chatgpt-4o-latest",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
} | GPT_5_MODELS

GPT_ALL_MODELS = (
    GPT_3_MODELS
    | GPT_3_16K_MODELS
    | GPT_4_MODELS
    | GPT_4_32K_MODELS
    | GPT_4_128K_MODELS
    | GPT_4O_MODELS
    | GPT_4_1_MODELS
    | GPT_5_MODELS
    | GPT_VISION_MODELS
    | O_MODELS
)


def default_max_tokens(model: str) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    base = 1200

    # Early returns for specific model categories
    if model in GPT_3_MODELS:
        return base

    if model in GPT_4_MODELS:
        return base * 2

    if model in GPT_3_16K_MODELS:
        return 4096 if model == "gpt-3.5-turbo-1106" else base * 4

    if model in GPT_4_32K_MODELS:
        return base * 8

    # All modern models use 4096 as default
    if model in (
        GPT_VISION_MODELS
        | GPT_4_128K_MODELS
        | GPT_4O_MODELS
        | GPT_4_1_MODELS
        | GPT_5_MODELS
        | O_MODELS
    ):
        return 4096

    raise ValueError(f"Unknown model: {model}")


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    # Early return for models that don't support functions
    non_function_models = {
        "gpt-3.5-turbo-0301",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
    }

    if model in non_function_models or model in O_MODELS:
        return False

    return True


# Load translations
current_file = Path(__file__)
translations_file_path = current_file.parent.parent / "translations.json"
with open(translations_file_path, encoding="utf-8") as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(
            f"No translation available for bot_language code '{bot_language}' and key '{key}'"
        )

        # Fallback to English if the translation is not available
        if key in translations["en"]:
            return translations["en"][key]

        # If no English translation exists, return the key itself
        logging.warning(
            f"No english definition found for key '{key}' in translations.json"
        )
        return key


def clean_latex_formatting(text: str) -> str:
    """
    Clean LaTeX formatting from text responses to make them more suitable for Telegram.
    Converts LaTeX math expressions to plain text format.
    """
    if not text:
        return text

    # Remove LaTeX inline math delimiters \( and \)
    text = re.sub(r"\\?\\\(", "", text)
    text = re.sub(r"\\?\\\)", "", text)

    # Remove LaTeX display math delimiters \[ and \]
    text = re.sub(r"\\?\\\[", "", text)
    text = re.sub(r"\\?\\\]", "", text)

    # Clean up common LaTeX commands that might appear
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)  # \text{...} -> ...
    text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", text)  # \mathrm{...} -> ...
    text = re.sub(
        r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", text
    )  # \frac{a}{b} -> (a)/(b)
    text = re.sub(r"\\cdot", "·", text)  # \cdot -> ·
    text = re.sub(r"\\times", "×", text)  # \times -> ×
    text = re.sub(r"\\div", "÷", text)  # \div -> ÷

    return text


class OpenAIHelper:
    """
    ChatGPT helper class.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager):
        """
        Initializes the OpenAI helper class with the given configuration.
        :param config: A dictionary containing the GPT configuration
        :param plugin_manager: The plugin manager
        """
        http_client = (
            httpx.AsyncClient(proxy=config["proxy"]) if "proxy" in config else None
        )
        self.client = openai.AsyncOpenAI(
            api_key=config["api_key"], http_client=http_client
        )
        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int, list] = {}  # {chat_id: history}
        self.conversations_vision: dict[int, bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int, datetime] = {}  # {chat_id: last_update_timestamp}

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)

        conversation = self.conversations[chat_id]
        return len(conversation), self.__count_tokens(conversation)

    def _ensure_conversation_exists(self, chat_id: int) -> None:
        """Ensure conversation exists and is not too old."""
        if chat_id not in self.conversations or self.__max_age_reached(chat_id):
            self.reset_chat_history(chat_id)
        self.last_updated[chat_id] = datetime.now()

    def _should_summarize_conversation(self, chat_id: int) -> bool:
        """Check if conversation should be summarized due to length limits."""
        token_count = self.__count_tokens(self.conversations[chat_id])
        exceeded_max_tokens = (
            token_count + self.config["max_tokens"] > self.__max_model_tokens()
        )
        exceeded_max_history_size = (
            len(self.conversations[chat_id]) > self.config["max_history_size"]
        )
        return exceeded_max_tokens or exceeded_max_history_size

    async def _handle_conversation_length(self, chat_id: int, query: str) -> None:
        """Handle conversation length by summarizing or truncating if needed."""
        if not self._should_summarize_conversation(chat_id):
            return

        logging.info(f"Chat history for chat ID {chat_id} is too long. Summarising...")

        try:
            # Try to summarize conversation
            summary = await self.__summarise(self.conversations[chat_id][:-1])
            logging.debug(f"Summary: {summary}")
            self.reset_chat_history(chat_id, self.conversations[chat_id][0]["content"])
            self.__add_to_history(chat_id, role="assistant", content=summary)
            self.__add_to_history(chat_id, role="user", content=query)
        except Exception as e:
            # Fallback to simple truncation
            logging.warning(
                f"Error while summarising chat history: {str(e)}. Popping elements instead..."
            )
            self.conversations[chat_id] = self.conversations[chat_id][
                -self.config["max_history_size"] :
            ]

    def _get_model_arguments(self, chat_id: int, stream: bool = False) -> dict:
        """Build common arguments for model requests."""
        max_tokens_str = (
            "max_completion_tokens"
            if self.config["model"] in O_MODELS
            else "max_tokens"
        )

        return {
            "model": (
                self.config["vision_model"]
                if self.conversations_vision[chat_id]
                else self.config["model"]
            ),
            "messages": self.conversations[chat_id],
            "temperature": self.config["temperature"],
            "n": self.config["n_choices"],
            max_tokens_str: self.config["max_tokens"],
            "presence_penalty": self.config["presence_penalty"],
            "frequency_penalty": self.config["frequency_penalty"],
            "stream": stream,
        }

    def _prepare_tools_for_request(self, chat_id: int) -> list:
        """Prepare tools/functions for the API request."""
        if not self.config["enable_functions"] or self.conversations_vision[chat_id]:
            return []

        function_specs = self.plugin_manager.get_functions_specs()
        return (
            self._convert_functions_to_tools(function_specs) if function_specs else []
        )

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, str]:
        """
        Gets a full response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query)
        if self.config["enable_functions"] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(
                chat_id, response
            )
            if is_direct_result(response):
                return response, "0"

        answer = ""

        # Handle Responses API output format

        answer = ""
        if hasattr(response, "output") and response.output:
            if isinstance(response.output, list) and len(response.output) > 0:
                output_item = response.output[0]
                if hasattr(output_item, "text"):
                    answer = output_item.text.strip()
                elif hasattr(output_item, "content"):
                    answer = output_item.content.strip()
                else:
                    answer = str(output_item).strip()
            else:
                if hasattr(response.output, "text"):
                    answer = response.output.text.strip()
                elif hasattr(response.output, "content"):
                    answer = response.output.content.strip()
                else:
                    answer = str(response.output).strip()
        elif hasattr(response, "output_text"):
            answer = response.output_text.strip()
        elif hasattr(response, "content"):
            answer = response.content.strip()
        else:
            # Fallback for compatibility
            logging.warning(
                "No recognizable content found in response, using str representation"
            )
            answer = str(response).strip()

        # Clean up LaTeX formatting for better Telegram display
        answer = clean_latex_formatting(answer)

        self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config["bot_language"]
        show_plugins_used = len(plugins_used) > 0 and self.config["show_plugins_used"]
        plugin_names = tuple(
            self.plugin_manager.get_plugin_source_name(plugin)
            for plugin in plugins_used
        )
        usage = self._get_usage_info(response)
        if self.config["show_usage"]:
            answer += (
                "\n\n---\n"
                f"💰 {str(usage.total_tokens)} {localized_text('stats_tokens', bot_language)}"
                f" ({str(usage.prompt_tokens)} {localized_text('prompt', bot_language)},"
                f" {str(usage.completion_tokens)} {localized_text('completion', bot_language)})"
            )
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, usage.total_tokens

    async def get_chat_response_stream(self, chat_id: int, query: str):
        """
        Stream response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, stream=True)
        if self.config["enable_functions"] and not self.conversations_vision[chat_id]:
            response, plugins_used = await self.__handle_function_call(
                chat_id, response, stream=True
            )
            if is_direct_result(response):
                yield response, "0"
                return

        answer = ""
        tokens_used = 0
        chunk_count = 0
        async for chunk in response:
            chunk_count += 1

            # Handle Responses API streaming format
            if hasattr(chunk, "type"):
                if chunk.type == "response.output_text.delta":
                    # This is a text delta from Responses API
                    if hasattr(chunk, "delta") and chunk.delta:
                        answer += chunk.delta
                        # Clean up LaTeX formatting in streaming content
                        cleaned_answer = clean_latex_formatting(answer)
                        yield cleaned_answer, "not_finished"
                elif chunk.type == "response.completed":
                    # Extract usage information from completed response
                    if hasattr(chunk, "response") and hasattr(chunk.response, "usage"):
                        tokens_used = chunk.response.usage.total_tokens
            elif hasattr(chunk, "output") and chunk.output:
                # Handle other Responses API events
                if isinstance(chunk.output, list) and len(chunk.output) > 0:
                    content = (
                        chunk.output[0].text
                        if hasattr(chunk.output[0], "text")
                        else str(chunk.output[0])
                    )
                    answer = content.strip()
                    # Clean up LaTeX formatting in streaming content
                    cleaned_answer = clean_latex_formatting(answer)
                    yield cleaned_answer, "not_finished"

        answer = answer.strip()
        if answer:  # Only add to history if we have content
            self.__add_to_history(chat_id, role="assistant", content=answer)

        # Fall back to token counting if we didn't get usage from the response
        if tokens_used == 0:
            tokens_used = self.__count_tokens(self.conversations[chat_id])

        tokens_used = str(tokens_used)

        # Clean up LaTeX formatting for better Telegram display (final answer)
        answer = clean_latex_formatting(answer)

        show_plugins_used = len(plugins_used) > 0 and self.config["show_plugins_used"]
        plugin_names = tuple(
            self.plugin_manager.get_plugin_source_name(plugin)
            for plugin in plugins_used
        )
        if self.config["show_usage"]:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, tokens_used

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
    )
    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config["bot_language"]
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.now()

            self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = (
                token_count + self.config["max_tokens"] > self.__max_model_tokens()
            )
            exceeded_max_history_size = (
                len(self.conversations[chat_id]) > self.config["max_history_size"]
            )

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(
                    f"Chat history for chat ID {chat_id} is too long. Summarising..."
                )
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f"Summary: {summary}")
                    self.reset_chat_history(
                        chat_id, self.conversations[chat_id][0]["content"]
                    )
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.__add_to_history(chat_id, role="user", content=query)
                except Exception as e:
                    logging.warning(
                        f"Error while summarising chat history: {str(e)}. Popping elements instead..."
                    )
                    self.conversations[chat_id] = self.conversations[chat_id][
                        -self.config["max_history_size"] :
                    ]

            max_tokens_str = (
                "max_completion_tokens"
                if self.config["model"] in O_MODELS
                else "max_tokens"
            )
            common_args = {
                "model": self.config["model"]
                if not self.conversations_vision[chat_id]
                else self.config["vision_model"],
                "messages": self.conversations[chat_id],
                "temperature": self.config["temperature"],
                "n": self.config["n_choices"],
                max_tokens_str: self.config["max_tokens"],
                "presence_penalty": self.config["presence_penalty"],
                "frequency_penalty": self.config["frequency_penalty"],
                "stream": stream,
            }

            # Convert messages format for Responses API
            input_messages = self._convert_messages_to_responses_format(
                self.conversations[chat_id]
            )

            # Prepare tools for Responses API
            tools = []
            if (
                self.config["enable_functions"]
                and not self.conversations_vision[chat_id]
            ):
                # Add custom plugin functions
                function_specs = self.plugin_manager.get_functions_specs()
                if len(function_specs) > 0:
                    tools.extend(self._convert_functions_to_tools(function_specs))

            # Create responses API arguments
            responses_args = {
                "model": common_args["model"],
                "input": input_messages,
                "stream": stream,
            }

            if tools:
                responses_args["tools"] = tools

            # Log request to OpenAI
            logging.info(
                f"Sending request to OpenAI API - Model: {responses_args['model']}, Tools: {len(tools) if tools else 0}"
            )

            result = await self.client.responses.create(**responses_args)
            return result

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(
                f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}"
            ) from e

        except Exception as e:
            raise Exception(
                f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}"
            ) from e

    async def __handle_function_call(
        self, chat_id, response, stream=False, times=0, plugins_used=()
    ):
        """
        Handle function calls for Responses API format.
        Responses API has a different structure - tool calls are handled differently.
        For now, return the response as-is since Responses API handles tools internally.
        """
        # Responses API handles tool calls internally, so we just return the response
        # This is a simplified approach - in production you might want to handle
        # tool calls more explicitly based on the response structure
        return response, plugins_used

    async def generate_image(self, prompt: str) -> tuple[str, str]:
        """
        Generates an image from the given prompt using DALL·E model.
        :param prompt: The prompt to send to the model
        :return: The image URL and the image size
        """
        bot_language = self.config["bot_language"]
        try:
            # Log request to OpenAI for image generation
            logging.info(
                f"Sending image generation request to OpenAI API - Model: {self.config['image_model']}, Size: {self.config['image_size']}"
            )

            response = await self.client.images.generate(
                prompt=prompt,
                n=1,
                model=self.config["image_model"],
                quality=self.config["image_quality"],
                style=self.config["image_style"],
                size=self.config["image_size"],
            )

            if len(response.data) == 0:
                logging.error(f"No response from GPT: {str(response)}")
                raise Exception(
                    f"⚠️ _{localized_text('error', bot_language)}._ "
                    f"⚠️\n{localized_text('try_again', bot_language)}."
                )

            return response.data[0].url, self.config["image_size"]
        except Exception as e:
            raise Exception(
                f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}"
            ) from e

    async def generate_speech(self, text: str) -> tuple[BytesIO, int]:
        """
        Generates an audio from the given text using TTS model.
        :param prompt: The text to send to the model
        :return: The audio in bytes and the text size
        """
        bot_language = self.config["bot_language"]
        try:
            # Log request to OpenAI for TTS
            logging.info(
                f"Sending TTS request to OpenAI API - Model: {self.config['tts_model']}, Voice: {self.config['tts_voice']}"
            )

            response = await self.client.audio.speech.create(
                model=self.config["tts_model"],
                voice=self.config["tts_voice"],
                input=text,
                response_format="opus",
            )

            temp_file = BytesIO()
            temp_file.write(response.read())
            temp_file.seek(0)
            return temp_file, len(text)
        except Exception as e:
            raise Exception(
                f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}"
            ) from e

    async def transcribe(self, filename):
        """
        Transcribes the audio file using the Whisper model.
        """
        try:
            with open(filename, "rb") as audio:
                prompt_text = self.config["whisper_prompt"]
                # Log request to OpenAI for transcription
                logging.info(
                    "Sending transcription request to OpenAI API - Model: whisper-1"
                )

                result = await self.client.audio.transcriptions.create(
                    model="whisper-1", file=audio, prompt=prompt_text
                )
                return result.text
        except Exception as e:
            logging.exception(e)
            raise Exception(
                f"⚠️ _{localized_text('error', self.config['bot_language'])}._ ⚠️\n{str(e)}"
            ) from e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
    )
    async def __common_get_chat_response_vision(
        self, chat_id: int, content: list, stream=False
    ):
        """
        Request a response from the GPT model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        bot_language = self.config["bot_language"]
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.now()

            if self.config["enable_vision_follow_up_questions"]:
                self.conversations_vision[chat_id] = True
                self.__add_to_history(chat_id, role="user", content=content)
            else:
                query = None
                for message in content:
                    if message.get("type") in ("text", "input_text"):
                        query = message.get("text", "")
                        break
                if query is None:
                    # Fallback: if no text part was found, use a default prompt
                    query = self.config.get("vision_prompt", "Describe the image")
                self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = (
                token_count + self.config["max_tokens"] > self.__max_model_tokens()
            )
            exceeded_max_history_size = (
                len(self.conversations[chat_id]) > self.config["max_history_size"]
            )

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(
                    f"Chat history for chat ID {chat_id} is too long. Summarising..."
                )
                try:
                    last = self.conversations[chat_id][-1]
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f"Summary: {summary}")
                    self.reset_chat_history(
                        chat_id, self.conversations[chat_id][0]["content"]
                    )
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.conversations[chat_id] += [last]
                except Exception as e:
                    logging.warning(
                        f"Error while summarising chat history: {str(e)}. Popping elements instead..."
                    )
                    self.conversations[chat_id] = self.conversations[chat_id][
                        -self.config["max_history_size"] :
                    ]

            message = {"role": "user", "content": content}

            common_args = {
                "model": self.config["vision_model"],
                "messages": self.conversations[chat_id][:-1] + [message],
                "temperature": self.config["temperature"],
                "n": 1,  # several choices is not implemented yet
                "max_tokens": self.config["vision_max_tokens"],
                "presence_penalty": self.config["presence_penalty"],
                "frequency_penalty": self.config["frequency_penalty"],
                "stream": stream,
            }

            # vision model does not yet support functions

            # if self.config['enable_functions']:
            #     functions = self.plugin_manager.get_functions_specs()
            #     if len(functions) > 0:
            #         common_args['functions'] = self.plugin_manager.get_functions_specs()
            #         common_args['function_call'] = 'auto'

            # Use standard Chat Completions API for vision
            vision_messages = self.conversations[chat_id][:-1] + [message]

            # Log a sanitized summary of the vision payload to aid debugging
            try:
                summary = []
                for idx, msg in enumerate(vision_messages):
                    kinds = []
                    parts = (
                        msg.get("content")
                        if isinstance(msg.get("content"), list)
                        else []
                    )
                    for p in parts:
                        if isinstance(p, dict):
                            t = p.get("type")
                            v = p.get("image_url") if "image_url" in p else None
                            vtype = type(v).__name__ if v is not None else None
                            kinds.append({"type": t, "image_url_type": vtype})
                    summary.append({"i": idx, "role": msg.get("role"), "kinds": kinds})
                logging.info(f"Vision messages summary: {summary}")
            except Exception:
                pass

            # Log request to OpenAI for vision
            logging.info(
                f"Sending vision request to OpenAI API - Model: {common_args['model']}"
            )

            return await self.client.chat.completions.create(
                model=common_args["model"],
                messages=vision_messages,
                temperature=common_args["temperature"],
                max_tokens=common_args["max_tokens"],
                presence_penalty=common_args["presence_penalty"],
                frequency_penalty=common_args["frequency_penalty"],
                stream=stream,
            )

        except openai.RateLimitError as e:
            raise e

        except openai.BadRequestError as e:
            raise Exception(
                f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}"
            ) from e

        except Exception as e:
            raise Exception(
                f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}"
            ) from e

    async def interpret_image(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config["vision_prompt"] if prompt is None else prompt

        # Build standard Chat Completions API content format for vision
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image}},
        ]

        response = await self.__common_get_chat_response_vision(chat_id, content)

        # functions are not available for this model

        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response)
        #     if is_direct_result(response):
        #         return response, '0'

        answer = ""

        # Handle standard Chat Completions API response format
        if hasattr(response, "choices") and response.choices:
            answer = response.choices[0].message.content.strip()
        else:
            # Fallback for compatibility
            answer = str(response).strip()

        self.__add_to_history(chat_id, role="assistant", content=answer)

        bot_language = self.config["bot_language"]
        # Plugins are not enabled either
        # show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        # plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)

        usage = response.usage if hasattr(response, "usage") else None
        total_tokens = usage.total_tokens if usage else 0

        if self.config["show_usage"] and usage:
            answer += (
                "\n\n---\n"
                f"💰 {str(usage.total_tokens)} {localized_text('stats_tokens', bot_language)}"
                f" ({str(usage.prompt_tokens)} {localized_text('prompt', bot_language)},"
                f" {str(usage.completion_tokens)} {localized_text('completion', bot_language)})"
            )
            # if show_plugins_used:
            #     answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        return answer, total_tokens

    async def interpret_image_stream(self, chat_id, fileobj, prompt=None):
        """
        Interprets a given PNG image file using the Vision model.
        """
        image = encode_image(fileobj)
        prompt = self.config["vision_prompt"] if prompt is None else prompt

        # Build standard Chat Completions API content format for vision
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image}},
        ]

        response = await self.__common_get_chat_response_vision(
            chat_id, content, stream=True
        )

        # if self.config['enable_functions']:
        #     response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
        #     if is_direct_result(response):
        #         yield response, '0'
        #         return

        answer = ""
        tokens_used = 0
        async for chunk in response:
            # Handle standard Chat Completions API streaming format
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and choice.delta:
                    if hasattr(choice.delta, "content") and choice.delta.content:
                        answer += choice.delta.content
                        yield answer, "not_finished"

                # Check for usage information (typically in the last chunk)
                if hasattr(chunk, "usage") and chunk.usage:
                    tokens_used = chunk.usage.total_tokens

        answer = answer.strip()
        if answer:  # Only add to history if we have content
            self.__add_to_history(chat_id, role="assistant", content=answer)

        # Fall back to token counting if we didn't get usage from the response
        if tokens_used == 0:
            tokens_used = self.__count_tokens(self.conversations[chat_id])

        tokens_used = str(tokens_used)

        # show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        # plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        if self.config["show_usage"]:
            answer += f"\n\n---\n💰 {tokens_used} {localized_text('stats_tokens', self.config['bot_language'])}"
        #     if show_plugins_used:
        #         answer += f"\n🔌 {', '.join(plugin_names)}"
        # elif show_plugins_used:
        #     answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"

        yield answer, tokens_used

    def reset_chat_history(self, chat_id, content=""):
        """
        Resets the conversation history.
        """
        if content == "":
            content = self.config["assistant_prompt"]
        self.conversations[chat_id] = [
            {
                "role": "assistant" if self.config["model"] in O_MODELS else "system",
                "content": content,
            }
        ]
        self.conversations_vision[chat_id] = False

    @staticmethod
    def _convert_messages_to_responses_format(messages: list) -> list:
        """
        Convert Chat Completions API messages format to Responses API input format.
        """
        # Responses API expects input in message format
        input_parts = []
        for message in messages:
            content = message["content"]

            # Handle vision messages with multimodal content
            if isinstance(content, list):
                converted_content = []
                for part in content:
                    if isinstance(part, dict) and "type" in part:
                        if part["type"] == "text":
                            converted_content.append(
                                {"type": "input_text", "text": part.get("text", "")}
                            )
                        elif part["type"] == "image_url":
                            image_field = part.get("image_url")
                            image_url = None
                            if isinstance(image_field, dict):
                                image_url = image_field.get("url")
                            elif isinstance(image_field, str):
                                image_url = image_field
                            # Build input_image part per Responses API expectations
                            image_part = {"type": "input_image"}
                            if image_url:
                                image_part["image_url"] = image_url
                            converted_content.append(image_part)
                        elif part["type"] in ("input_text", "input_image"):
                            # Already in Responses API format; keep as-is
                            converted_content.append(part)
                        else:
                            # Keep other types as-is
                            converted_content.append(part)
                    else:
                        converted_content.append(part)
                content = converted_content

            input_parts.append(
                {"type": "message", "role": message["role"], "content": content}
            )
        return input_parts

    @staticmethod
    def _convert_functions_to_tools(function_specs: list) -> list:
        """
        Convert function specifications from Chat Completions format to Responses API tools format.
        """
        tools = []
        for func_spec in function_specs:
            # Responses API expects the function spec directly with required fields
            tool = {
                "type": "function",
                "name": func_spec["name"],
                "description": func_spec.get("description", ""),
                "parameters": func_spec.get("parameters", {}),
            }
            tools.append(tool)
        return tools

    def _get_usage_info(self, response):
        """
        Extract usage information from Responses API response.
        """
        if hasattr(response, "usage"):
            return response.usage
        elif hasattr(response, "metadata") and hasattr(response.metadata, "usage"):
            return response.metadata.usage
        else:
            # Fallback with dummy usage for compatibility
            class DummyUsage:
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0

            return DummyUsage()

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.now()
        max_age_minutes = self.config["max_conversation_age_minutes"]
        return last_updated < now - timedelta(minutes=max_age_minutes)

    def __add_function_call_to_history(self, chat_id, function_name, content):
        """
        Adds a function call to the conversation history
        """
        self.conversations[chat_id].append(
            {"role": "function", "name": function_name, "content": content}
        )

    def __add_to_history(self, chat_id, role, content):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :return: The summary
        """
        input_content = [
            {
                "type": "message",
                "role": "system",
                "content": "Summarize this conversation in 700 characters or less",
            },
            {"type": "message", "role": "user", "content": str(conversation)},
        ]
        # Log request to OpenAI for conversation summary
        logging.info(
            f"Sending summary request to OpenAI API - Model: {self.config['model']}"
        )

        response = await self.client.responses.create(
            model=self.config["model"],
            input=input_content,
        )
        return (
            response.output[0].text
            if hasattr(response, "output")
            else response.output_text
        )

    def __max_model_tokens(self):
        base = 4096
        if self.config["model"] in GPT_3_MODELS:
            return base
        if self.config["model"] in GPT_3_16K_MODELS:
            return base * 4
        if self.config["model"] in GPT_4_MODELS:
            return base * 2
        if self.config["model"] in GPT_4_32K_MODELS:
            return base * 8
        if self.config["model"] in GPT_VISION_MODELS:
            return base * 31
        if self.config["model"] in GPT_4_128K_MODELS:
            return base * 31
        if self.config["model"] in GPT_4O_MODELS:
            return base * 31
        if self.config["model"] in GPT_4_1_MODELS:
            return base * 31
        if self.config["model"] in GPT_5_MODELS:
            return base * 31
        elif self.config["model"] in O_MODELS:
            # https://platform.openai.com/docs/models#o1
            if self.config["model"] == "o1":
                return 100_000
            elif self.config["model"] == "o1-preview":
                return 32_768
            else:
                return 65_536
        raise NotImplementedError(
            f"Max tokens for model {self.config['model']} is not implemented yet."
        )

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def __count_tokens(self, messages) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        model = self.config["model"]
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")

        if model in GPT_ALL_MODELS:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == "content":
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value))
                    else:
                        for message1 in value:
                            mtype = (
                                message1.get("type")
                                if isinstance(message1, dict)
                                else None
                            )
                            if mtype in ("image_url", "input_image"):
                                # Support both legacy Chat Completions-style and Responses-style
                                url_val = None
                                if mtype == "image_url":
                                    # legacy format: {"image_url": {"url": "data:..."}}
                                    inner = message1.get("image_url")
                                    if isinstance(inner, dict):
                                        url_val = inner.get("url")
                                    elif isinstance(inner, str):
                                        url_val = inner
                                else:
                                    # input_image format: {"image_url": "data:..."}
                                    url_val = message1.get("image_url")
                                if url_val:
                                    image = decode_image(url_val)
                                    num_tokens += self.__count_tokens_vision(image)
                            else:
                                text_val = (
                                    message1.get("text", "")
                                    if isinstance(message1, dict)
                                    else str(message1)
                                )
                                num_tokens += len(encoding.encode(text_val))
                else:
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    # no longer needed

    def __count_tokens_vision(self, image_bytes: bytes) -> int:
        """
        Counts the number of tokens for interpreting an image.
        :param image_bytes: image to interpret
        :return: the number of tokens required
        """
        image_file = BytesIO(image_bytes)
        image = Image.open(image_file)
        model = self.config["vision_model"]
        if model not in GPT_VISION_MODELS:
            raise NotImplementedError(
                f"""count_tokens_vision() is not implemented for model {model}."""
            )

        w, h = image.size
        if w > h:
            w, h = h, w
        # this computation follows https://platform.openai.com/docs/guides/vision and https://openai.com/pricing#gpt-4-turbo
        base_tokens = 85
        detail = self.config["vision_detail"]
        if detail == "low":
            return base_tokens
        elif detail == "high" or detail == "auto":  # assuming worst cost for auto
            f = max(w / 768, h / 2048)
            if f > 1:
                w, h = int(w / f), int(h / f)
            tw, th = (w + 511) // 512, (h + 511) // 512
            tiles = tw * th
            num_tokens = base_tokens + tiles * 170
            return num_tokens
        else:
            raise NotImplementedError(
                f"""unknown parameter detail={detail} for model {model}."""
            )
