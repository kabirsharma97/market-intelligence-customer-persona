import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="twinsim_ui",
    frontend_port=3000,
    backend_port=8000,
    plugins=[SitemapPlugin()],
)
