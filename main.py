"""Entrypoint to launch the FastAPI server directly."""

import uvicorn


def main():
    print("Starting Sistem Deteksi APD PDU Migas Backend Server on http://0.0.0.0:8000 ...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
