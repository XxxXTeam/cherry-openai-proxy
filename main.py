import logging

from app import create_app
from app.config import Settings


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


configure_logging()
settings = Settings.from_env()
app = create_app(settings)


def main() -> None:
    app.run(host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
