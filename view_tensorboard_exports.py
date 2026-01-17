#!/usr/bin/env python3
"""
Simple HTML viewer generator for exported TensorBoard images.
Usage: python view_tensorboard_exports.py [--export-dir PATH]
"""

import argparse
from pathlib import Path
from collections import defaultdict


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TensorBoard Exports - {title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #1e1e1e;
            color: #e0e0e0;
            padding: 20px;
        }}

        .header {{
            background: #2d2d30;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}

        h1 {{
            color: #4ec9b0;
            margin-bottom: 10px;
        }}

        .stats {{
            color: #808080;
            font-size: 14px;
        }}

        .navigation {{
            background: #2d2d30;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .nav-button {{
            background: #3e3e42;
            color: #e0e0e0;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}

        .nav-button:hover {{
            background: #505053;
        }}

        .nav-button.active {{
            background: #0e639c;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section-title {{
            color: #569cd6;
            font-size: 24px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3e3e42;
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
        }}

        .image-card {{
            background: #2d2d30;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4);
        }}

        .image-title {{
            color: #dcdcaa;
            font-size: 14px;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .image-container {{
            position: relative;
            overflow: hidden;
            border-radius: 4px;
            background: #1e1e1e;
        }}

        .image-container img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
            transition: transform 0.3s;
        }}

        .image-container img:hover {{
            transform: scale(1.02);
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            overflow: auto;
        }}

        .modal-content {{
            margin: 2% auto;
            display: block;
            max-width: 95%;
            max-height: 95%;
        }}

        .modal-close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}

        .modal-close:hover {{
            color: #ff4444;
        }}

        .comparison {{
            margin-bottom: 30px;
        }}

        .comparison img {{
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }}

        @media (max-width: 768px) {{
            .image-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 TensorBoard Exports</h1>
        <div class="stats">{stats}</div>
    </div>

    <div class="navigation">
        {navigation}
    </div>

    {comparison_section}

    {sections}

    <div id="imageModal" class="modal" onclick="this.style.display='none'">
        <span class="modal-close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        // Image modal
        document.querySelectorAll('.image-container img').forEach(img => {{
            img.onclick = function() {{
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                modal.style.display = 'block';
                modalImg.src = this.src;
            }};
        }});

        // Navigation
        document.querySelectorAll('.nav-button').forEach(button => {{
            button.onclick = function() {{
                const targetId = this.dataset.target;
                document.getElementById(targetId).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }};
        }});
    </script>
</body>
</html>
"""


def generate_html(export_dir):
    """Generate HTML viewer from exported images."""
    export_path = Path(export_dir)

    if not export_path.exists():
        print(f"Error: {export_dir} does not exist")
        return None

    # Find all images
    image_files = list(export_path.glob("*.png"))

    if not image_files:
        print(f"No images found in {export_dir}")
        return None

    # Organize images by category
    categories = defaultdict(list)
    comparison_image = None

    for img_path in image_files:
        img_name = img_path.name

        if img_name in ['comparison_plot.png', 'metrics_grid.png']:
            comparison_image = img_name
        else:
            # Extract category from filename
            if '_' in img_name:
                category = img_name.split('_')[0]
            else:
                category = 'Other'

            categories[category].append(img_name)

    # Sort categories and images
    sorted_categories = sorted(categories.items())

    # Generate navigation
    nav_html = ""
    if comparison_image:
        nav_html += '<button class="nav-button" data-target="comparison">📊 Comparison</button>'

    for category, _ in sorted_categories:
        nav_html += f'<button class="nav-button" data-target="section-{category}">{category}</button>'

    # Generate comparison section
    comparison_html = ""
    if comparison_image:
        comparison_html = f"""
    <div id="comparison" class="comparison">
        <div class="section-title">📊 Comparison Overview</div>
        <img src="{comparison_image}" alt="Comparison Plot" style="cursor: pointer;"
             onclick="document.getElementById('imageModal').style.display='block';
                      document.getElementById('modalImage').src=this.src;">
    </div>
        """

    # Generate sections
    sections_html = ""
    for category, images in sorted_categories:
        sections_html += f"""
    <div id="section-{category}" class="section">
        <div class="section-title">{category} Metrics</div>
        <div class="image-grid">
        """

        for img_name in sorted(images):
            # Clean title
            title = img_name.replace('.png', '').replace('_', ' ').title()

            sections_html += f"""
            <div class="image-card">
                <div class="image-title">{title}</div>
                <div class="image-container">
                    <img src="{img_name}" alt="{title}">
                </div>
            </div>
            """

        sections_html += """
        </div>
    </div>
        """

    # Generate stats
    total_images = len(image_files)
    stats = f"Total Images: {total_images} | Categories: {len(categories)}"

    # Fill template
    html_content = HTML_TEMPLATE.format(
        title=export_path.name,
        stats=stats,
        navigation=nav_html,
        comparison_section=comparison_html,
        sections=sections_html
    )

    # Write HTML file
    output_path = export_path / "index.html"
    output_path.write_text(html_content)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML viewer for TensorBoard exports'
    )
    parser.add_argument('--export-dir', type=str, default='tensorboard_exports',
                        help='Directory containing exported images')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TensorBoard Export Viewer Generator")
    print("="*60 + "\n")

    output_path = generate_html(args.export_dir)

    if output_path:
        print(f"✓ HTML viewer generated: {output_path.absolute()}")
        print(f"\nOpen in browser:")
        print(f"  file://{output_path.absolute()}")
        print()


if __name__ == "__main__":
    main()
