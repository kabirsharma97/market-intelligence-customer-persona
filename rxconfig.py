import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="twinsim_ui",
    frontend_port=3000,
    backend_port=8000,
    plugins=[SitemapPlugin()],
    # Allow uploads up to 50 MB (session CSV is ~7 MB, headroom for growth)
    upload_max_file_size=50_000_000,
)
