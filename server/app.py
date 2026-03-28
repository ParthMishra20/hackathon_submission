"""OpenEnv server entrypoint."""

import uvicorn


def main() -> None:
    """Run the OpenEnv Support Triage server."""
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
