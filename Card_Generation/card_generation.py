from PIL import Image, ImageDraw, ImageFont
import io
import textwrap
import matplotlib.pyplot as plt

def create_card(plots, title, description, output_filename="poster.png"):
    """
    Function to create a poster with provided plots and text using Pillow.
    """

    # Step 1: Create a Blank Canvas (A smaller canvas to reduce memory usage)
    width, height = 1240, 1754  # Smaller A4 size at 150 dpi
    background_color = (255, 255, 255)  # White background
    poster = Image.new('RGB', (width, height), background_color)

    # Step 2: Add Text (Title and Description)
    draw = ImageDraw.Draw(poster)

    # Use default fonts to minimize memory usage
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    # Wrap the title and description to fit within poster width
    wrapped_title = textwrap.fill(title, width=30)
    wrapped_description = textwrap.fill(description, width=60)

    # Draw title and description text in chunks to prevent memory overload
    draw.text((100, 50), wrapped_title, font=font_title, fill="black")
    draw.text((100, 300), wrapped_description, font=font_body, fill="black")

    # Step 3: Convert Matplotlib Plots to Images and Paste on the Canvas
    plot_width, plot_height = 800, 600  # Smaller plot dimensions
    for i, plot in enumerate(plots):
        img_buf = io.BytesIO()
        plot.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        plot_image = Image.open(img_buf)

        # Resize the plot image to fit the poster
        plot_image.thumbnail((plot_width, plot_height), Image.Resampling.LANCZOS)

        # Calculate positions for each plot (stack vertically)
        y_position = 500 + i * (plot_height + 50)
        poster.paste(plot_image, (100, y_position))

        # Get the figure from the plot and close it to free memory
        # plt.close(plot)

    # Step 4: Save the Final Poster as PNG or PDF
    poster.save(output_filename, optimize=True)  # Optimize the image
    print(f"Poster saved as {output_filename}")
