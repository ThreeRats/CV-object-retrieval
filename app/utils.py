import base64
from io import BytesIO
from PIL import Image

def Base64Encoding(image: Image.Image) -> str:
    """
    将Image.Image图片加密成为base64 str
    """
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    byte_data = buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return str(base64_str)[2:-1]

def Base64Decoding(base64_str: str, show: bool = False) -> Image.Image:
    """
    decoding base64,转换为一张图片
    """
    byte_data = base64.b64decode(base64_str)
    image_data = BytesIO(byte_data)
    original_img = Image.open(image_data).convert('RGB')

    if show:
        original_img.show()

    return original_img
