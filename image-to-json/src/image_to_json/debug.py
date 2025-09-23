import io
from PIL import Image
import requests

def get_image(url):
    r = requests.get(url)
    return Image.open(io.BytesIO(r.content))

image = get_image("https://picsum.photos/id/237/400/300")
print(type(image))