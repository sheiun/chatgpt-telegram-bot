import whois

from .plugin import Plugin


class WhoisPlugin(Plugin):
    """
    A plugin to query whois database
    """

    def get_source_name(self) -> str:
        return "Whois"

    def get_spec(self) -> list[dict]:
        return [
            {
                "name": "get_whois",
                "description": "Get whois registration and expiry information for a domain",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "description": "Domain name"}
                    },
                    "required": ["domain"],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> dict:
        try:
            whois_result = whois.query(kwargs["domain"])
            if whois_result is None:
                return {"result": "No such domain found"}
            return whois_result.__dict__
        except Exception as e:
            return {"error": "An unexpected error occurred: " + str(e)}
