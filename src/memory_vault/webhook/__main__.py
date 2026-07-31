
from . import serve

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory Vault Webhook")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9876, help="Port (default: 9876)")
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
