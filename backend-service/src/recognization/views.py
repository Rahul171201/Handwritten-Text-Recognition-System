import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import base64

from .utils import *

# IMAGE RECOGNITION-----------------------------------------------------------------------------------

@api_view(['POST'])
def image_recognition(request):
    try:
        response = {"Error_data": status.HTTP_200_OK, "message": "no error"}
        # Load inputs
        inputs = request.data
        image = inputs['image']

        # Decode base64 image
        image = base64.b64decode(image)
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape(28, 28)

        # # call image predict function
        # prediction = image_predict(image)
        # response['message'] = prediction

        return JsonResponse(response, safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
        
# IMAGE RECOGNITION-----------------------------------------------------------------------------------
