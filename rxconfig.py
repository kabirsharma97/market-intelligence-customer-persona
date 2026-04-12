import os
import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

# Raise WebSocket buffer to 50 MB so large CSVs (session events ~7 MB) upload correctly
os.environ.setdefault("REFLEX_SOCKET_MAX_HTTP_BUFFER_SIZE", str(50 * 1024 * 1024))

config = rx.Config(
    app_name="twinsim_ui",
    frontend_port=3000,
    backend_port=8000,
    plugins=[SitemapPlugin()],
)
