from flask import Request


def is_authorized(request: Request, expected_api_key: str | None) -> bool:
    if not expected_api_key:
        return True

    header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        return False

    token = header[len(prefix) :].strip()
    return token == expected_api_key
