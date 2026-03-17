"""nQPU QKD API entry point.

Launch via::

    python -m nqpu.web.app          # or
    uvicorn nqpu.web.qkd_api:app    # for development with --reload
"""


def main() -> None:
    """Run the QKD Network Planner API server."""
    import uvicorn

    from nqpu.web.qkd_api import app

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
