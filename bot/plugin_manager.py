import json
import logging
from typing import Any

from plugins.auto_tts import AutoTextToSpeech
from plugins.crypto import CryptoPlugin
from plugins.ddg_image_search import DDGImageSearchPlugin
from plugins.ddg_web_search import DDGWebSearchPlugin
from plugins.deepl import DeeplTranslatePlugin
from plugins.dice import DicePlugin
from plugins.gtts_text_to_speech import GTTSTextToSpeech
from plugins.iplocation import IpLocationPlugin
from plugins.spotify import SpotifyPlugin
from plugins.weather import WeatherPlugin
from plugins.webshot import WebshotPlugin
from plugins.whois_ import WhoisPlugin
from plugins.wolfram_alpha import WolframAlphaPlugin
from plugins.worldtimeapi import WorldTimeApiPlugin
from plugins.youtube_audio_extractor import YouTubeAudioExtractorPlugin


class PluginManager:
    """
    A class to manage the plugins and call the correct functions
    """

    def __init__(self, config: dict[str, Any]):
        self.plugins = self._initialize_plugins(config)

    def _get_plugin_mapping(self) -> dict[str, type]:
        """
        Returns the mapping of plugin names to their classes.
        """
        return {
            "wolfram": WolframAlphaPlugin,
            "weather": WeatherPlugin,
            "crypto": CryptoPlugin,
            "ddg_web_search": DDGWebSearchPlugin,
            "ddg_image_search": DDGImageSearchPlugin,
            "spotify": SpotifyPlugin,
            "worldtimeapi": WorldTimeApiPlugin,
            "youtube_audio_extractor": YouTubeAudioExtractorPlugin,
            "dice": DicePlugin,
            "deepl_translate": DeeplTranslatePlugin,
            "gtts_text_to_speech": GTTSTextToSpeech,
            "auto_tts": AutoTextToSpeech,
            "whois": WhoisPlugin,
            "webshot": WebshotPlugin,
            "iplocation": IpLocationPlugin,
        }

    def _initialize_plugins(self, config: dict[str, Any]) -> list[Any]:
        """
        Initialize plugins based on configuration.
        """
        enabled_plugins = config.get("plugins", [])
        plugin_mapping = self._get_plugin_mapping()

        plugins = []
        for plugin_name in enabled_plugins:
            # Skip empty plugin names
            if not plugin_name or not plugin_name.strip():
                continue

            plugin_name = plugin_name.strip()
            if plugin_name in plugin_mapping:
                try:
                    plugin_instance = plugin_mapping[plugin_name]()
                    plugins.append(plugin_instance)
                    logging.info(f"Successfully loaded plugin: {plugin_name}")
                except Exception as e:
                    logging.error(f"Failed to load plugin {plugin_name}: {e}")
            else:
                logging.warning(f"Unknown plugin: {plugin_name}")

        return plugins

    def get_functions_specs(self) -> list[dict[str, Any]]:
        """
        Return the list of function specs that can be called by the model
        """
        return [spec for plugin in self.plugins for spec in plugin.get_spec()]

    async def call_function(
        self, function_name: str, helper: Any, arguments: str
    ) -> str:
        """
        Call a function based on the name and parameters provided
        """
        plugin = self._get_plugin_by_function_name(function_name)
        if not plugin:
            return json.dumps({"error": f"Function {function_name} not found"})

        try:
            parsed_arguments = json.loads(arguments)
            result = await plugin.execute(function_name, helper, **parsed_arguments)
            return json.dumps(result, default=str)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON arguments: {e}"})
        except Exception as e:
            logging.error(f"Error executing function {function_name}: {e}")
            return json.dumps({"error": f"Function execution failed: {e}"})

    def get_plugin_source_name(self, function_name: str) -> str:
        """
        Return the source name of the plugin
        """
        plugin = self._get_plugin_by_function_name(function_name)
        if not plugin:
            return ""
        return plugin.get_source_name()

    def _get_plugin_by_function_name(self, function_name: str) -> Any | None:
        """
        Find a plugin that provides the specified function.
        """
        for plugin in self.plugins:
            function_names = [spec.get("name") for spec in plugin.get_spec()]
            if function_name in function_names:
                return plugin
        return None
