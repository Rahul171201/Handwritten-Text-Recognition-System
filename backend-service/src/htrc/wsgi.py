import os

from django.core.wsgi import get_wsgi_application

ServiceRunMode = os.getenv('ServiceRunMode')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', ServiceRunMode)

application = get_wsgi_application()
