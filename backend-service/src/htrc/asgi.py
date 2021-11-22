import os

from django.core.asgi import get_asgi_application

ServiceRunMode = os.getenv('ServiceRunMode')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', ServiceRunMode)

application = get_asgi_application()
