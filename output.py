from PIL import Image, ImageDraw


class PrettyPrint:
    @staticmethod
    def print_pretty(image_data):
        width = 28
        height = 28

        image = Image.new("RGB", (width, height), "white")

        draw = ImageDraw.Draw(image)

        colors = {
            '.': (255, 255, 255),  # White
            '@': (255, 0, 0),  # Red
        }

        for y, line in enumerate(image_data.strip().split('\n')):
            for x, char in enumerate(line):
                if char in colors:
                    draw.point((x, y), fill=colors[char])

        image.show()
