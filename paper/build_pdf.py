import re
import os
import subprocess
import sys
import html
import urllib.request

def code_to_html(code_text):
    if code_text.startswith("```"):
        # Fenced code block
        content = code_text.strip("`").strip("\n")
        # Remove optional language specifier if present (e.g. python, bash)
        lines = content.split("\n")
        if lines and lines[0].strip() in ["python", "bash", "javascript", "json", "html", "css", "markdown", "mermaid"]:
            lines = lines[1:]
        content = "\n".join(lines)
        escaped = html.escape(content)
        return f"<pre><code>{escaped}</code></pre>"
    else:
        # Inline code
        content = code_text.strip("`")
        escaped = html.escape(content)
        return f"<code>{escaped}</code>"

def get_katex_css():
    url = "https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"
    try:
        print("Fetching KaTeX CSS from CDN to inline...")
        with urllib.request.urlopen(url, timeout=5) as response:
            css = response.read().decode('utf-8')
            # Replace relative font paths with absolute CDN paths
            css = css.replace("url(fonts/", "url(https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/fonts/")
            print("Successfully fetched and resolved KaTeX CSS.")
            return f"<style>{css}</style>"
    except Exception as e:
        print(f"Warning: Failed to fetch KaTeX CSS from CDN: {e}. Falling back to CDN link.")
        return '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">'

def build():
    print("Starting Modus whitepaper PDF build pipeline...")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if whitepaper.md exists
    md_path = os.path.join(base_dir, "whitepaper.md")
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        sys.exit(1)
        
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("File loaded. Protecting code blocks and LaTeX math...")
    
    # 1. Protect code blocks (fenced and inline) so markdown parser doesn't touch anything inside them
    code_blocks = []
    def placeholder_code(match):
        code_blocks.append(match.group(0))
        return f"CODEBLOCKPLACEHOLDER{len(code_blocks)-1}"
    # Fenced code blocks
    text = re.sub(r"```.*?```", placeholder_code, text, flags=re.DOTALL)
    # Inline code blocks
    text = re.sub(r"`[^`\n]+?`", placeholder_code, text)
    
    # 2. Protect display math $$...$$ and inline math $...$
    math_blocks = []
    def placeholder_math(match):
        math_blocks.append(match.group(0))
        return f"MATHBLOCKPLACEHOLDER{len(math_blocks)-1}"
    # Display math
    text = re.sub(r"\$\$.*?\$\$", placeholder_math, text, flags=re.DOTALL)
    # Inline math (only letters/math characters inside, to avoid false positives)
    text = re.sub(r"\$([^\$\n]+?)\$", placeholder_math, text)
    
    # 3. Extract Metadata (Title, Authors, Abstract)
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title = title_match.group(1) if title_match else "Modus_X: Dual-Stream Hybrid Language Modeling"
    
    abstract_match = re.search(r"## Abstract\n\n(.*?)\n\n---", text, re.DOTALL)
    abstract_text = abstract_match.group(1) if abstract_match else ""
    
    # Locate the start of Section 1 (Introduction) and strip everything before it (like Table of Contents)
    body_start_idx = text.find("## 1. Introduction")
    if body_start_idx == -1:
        # Fallback if Introduction title has slightly different markup
        body_start_idx = text.find("---", text.find("## Abstract"))
        if body_start_idx != -1:
            body_start_idx += 3
        else:
            body_start_idx = 0
            
    body_text = text[body_start_idx:]
    print("Parsing Markdown body to HTML...")
    
    # Import markdown package
    import markdown
    html_body = markdown.markdown(body_text, extensions=['tables', 'fenced_code'])
    
    # 4. Restore math blocks (in descending order to prevent substring/prefix collisions)
    for i in sorted(range(len(math_blocks)), reverse=True):
        html_body = html_body.replace(f"MATHBLOCKPLACEHOLDER{i}", math_blocks[i])
        
    # 5. Restore code blocks (in descending order to prevent substring/prefix collisions)
    for i in sorted(range(len(code_blocks)), reverse=True):
        code_html = code_to_html(code_blocks[i])
        if code_blocks[i].startswith("```"):
            html_body = html_body.replace(f"<p>CODEBLOCKPLACEHOLDER{i}</p>", f"CODEBLOCKPLACEHOLDER{i}")
        html_body = html_body.replace(f"CODEBLOCKPLACEHOLDER{i}", code_html)
        
    # Restore abstract if present
    if abstract_text:
        # Compile abstract markdown to HTML
        abstract_text = markdown.markdown(abstract_text, extensions=['tables', 'fenced_code'])
        # Restore abstract math & code blocks as well (in descending order)
        for i in sorted(range(len(math_blocks)), reverse=True):
            abstract_text = abstract_text.replace(f"MATHBLOCKPLACEHOLDER{i}", math_blocks[i])
        for i in sorted(range(len(code_blocks)), reverse=True):
            code_html = code_to_html(code_blocks[i])
            if code_blocks[i].startswith("```"):
                abstract_text = abstract_text.replace(f"<p>CODEBLOCKPLACEHOLDER{i}</p>", f"CODEBLOCKPLACEHOLDER{i}")
            abstract_text = abstract_text.replace(f"CODEBLOCKPLACEHOLDER{i}", code_html)
            
    # Fetch and inline KaTeX CSS for robust rendering
    katex_css_block = get_katex_css()
    print("Constructing styled HTML template...")
    
    # Style template matching premium LaTeX / ACM journals
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    
    <!-- Premium Fonts -->
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@400;500;600;700&display=swap">
    
    <!-- KaTeX for instantaneous, synchronous LaTeX Math rendering -->
    {katex_css_block}
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
    
    <!-- Mermaid CDN to support flowcharts -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    
    <style>
        @page {{
            size: letter;
            margin: 24mm 24mm 24mm 24mm;
        }}
        
        body {{
            font-family: 'EB Garamond', Georgia, serif;
            font-size: 11.5pt;
            line-height: 1.55;
            color: #1a1a1a;
            background: #ffffff;
            margin: 0;
            padding: 0;
            text-align: justify;
            text-justify: inter-word;
        }}
        
        /* Centered Premium Title */
        .title-container {{
            text-align: center;
            margin-top: 1em;
            margin-bottom: 2.2em;
            page-break-after: avoid;
        }}
        
        .paper-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 24pt;
            font-weight: 700;
            line-height: 1.25;
            color: #111111;
            margin-bottom: 0.5em;
        }}
        
        .authors {{
            font-family: 'Outfit', sans-serif;
            font-size: 10.5pt;
            color: #4a5568;
            line-height: 1.5;
        }}
        
        .authors strong {{
            font-size: 11.5pt;
            color: #111111;
        }}
        
        /* Abstract Block */
        .abstract-box {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #4a5568;
            border-radius: 4px;
            padding: 18px 24px;
            margin: 2em auto;
            width: 95%;
            font-style: italic;
            font-size: 10pt;
            box-sizing: border-box;
            line-height: 1.6;
            page-break-inside: avoid;
        }}
        
        .abstract-title {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-style: normal;
            text-transform: uppercase;
            font-size: 9pt;
            letter-spacing: 1px;
            color: #2d3748;
            margin-bottom: 0.6em;
            text-align: center;
        }}
        
        /* Headers styling */
        h2, h3, h4 {{
            font-family: 'Outfit', sans-serif;
            color: #111111;
            font-weight: 700;
            text-align: left;
            margin-top: 1.8em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }}
        
        h2 {{
            font-size: 14pt;
            border-bottom: 1.2px solid #e2e8f0;
            padding-bottom: 4px;
            margin-top: 2em;
        }}
        
        h3 {{
            font-size: 12pt;
            margin-top: 1.6em;
        }}
        
        h4 {{
            font-size: 11pt;
            font-style: italic;
            font-weight: 600;
        }}
        
        /* Paragraph spacing & indent */
        p {{
            margin-top: 0;
            margin-bottom: 0.8em;
            text-indent: 1.5em;
        }}
        
        /* Do not indent first paragraph of a section or after blocks */
        p:first-of-type, h2 + p, h3 + p, h4 + p, ul + p, ol + p, div + p, table + p, pre + p, blockquote + p {{
            text-indent: 0;
        }}
        
        /* LaTeX Booktabs Tables */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 2em auto;
            font-size: 10pt;
            font-family: 'Outfit', sans-serif;
            page-break-inside: avoid;
        }}
        
        th, td {{
            padding: 8px 14px;
            text-align: left;
        }}
        
        th {{
            border-top: 1.8px solid #111;
            border-bottom: 1.1px solid #111;
            font-weight: 600;
            color: #111;
            background-color: #f8fafc;
        }}
        
        td {{
            border-bottom: 0.6px solid #e2e8f0;
        }}
        
        tr:last-child td {{
            border-bottom: 1.8px solid #111;
        }}
        
        /* Code blocks */
        pre {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            padding: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 9pt;
            line-height: 1.45;
            overflow-x: auto;
            margin: 1.8em 0;
            page-break-inside: avoid;
        }}
        
        code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 9pt;
            background-color: #f8fafc;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            border-radius: 0;
        }}
        
        /* Images and figures */
        p img, div img {{
            display: block;
            max-width: 95%;
            height: auto;
            margin: 2.2em auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid #eaeaea;
            page-break-inside: avoid;
        }}
        
        em, i, _figure_caption_ {{
            font-family: 'EB Garamond', Georgia, serif;
        }}
        
        /* Let paragraph italics stand out beautifully */
        p em, li em {{
            font-style: italic;
        }}
        
        /* Custom formatting for math expressions in text */
        .katex {{
            font-size: 1.05em !important;
        }}
        
        /* Lists formatting */
        ul, ol {{
            margin-top: 0;
            margin-bottom: 1em;
            padding-left: 24px;
            text-align: left;
        }}
        
        li {{
            margin-bottom: 0.5em;
            text-align: justify;
        }}
        
        /* Section dividers */
        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 2.5em 0;
        }}
        
        /* Convert mermaid elements into valid container blocks */
        .mermaid {{
            margin: 2em auto;
            text-align: center;
            page-break-inside: avoid;
        }}
    </style>
</head>
<body>
    <!-- Center LaTeX Title -->
    <div class="title-container">
        <div class="paper-title">{title}</div>
        <div class="authors">
            <strong>Sanyam Chaudhary</strong><br>
            Independent Researcher, India<br>
            <span style="color: #718096; font-size: 9.5pt;">June 2026 &nbsp;|&nbsp; Modus Research Project</span><br>
            <span style="margin-top: 5px; display: inline-block; font-size: 9.5pt; font-family: 'JetBrains Mono', monospace;"><a href="https://doi.org/10.5281/zenodo.20443699" style="color: #4a5568; text-decoration: none; border-bottom: 1px solid #cbd5e1;">doi.org/10.5281/zenodo.20443699</a> &nbsp;|&nbsp; <a href="https://github.com/sanyamChaudhary27/Modus_X" style="color: #4a5568; text-decoration: none; border-bottom: 1px solid #cbd5e1;">github.com/sanyamChaudhary27/Modus_X</a></span>
        </div>
    </div>
    <!-- Centered Abstract -->
    <div class="abstract-box">
        <div class="abstract-title">Abstract</div>
        {abstract_text}
    </div>
    <!-- Main HTML Body -->
    {html_body}
    <!-- Instant Auto-Render of LaTeX Math Equations -->
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            renderMathInElement(document.body, {{
                delimiters: [
                    {{left: "$$", right: "$$", display: true}},
                    {{left: "$", right: "$", display: false}}
                ],
                throwOnError: false
            }});
        }});
    </script>
</body>
</html>
"""
    html_path = os.path.join(base_dir, "whitepaper.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Beautiful publication-grade HTML output generated successfully at: {html_path}")
    
    # 6. Call headless Chrome to render and print to PDF!
    print("Calling headless Google Chrome for PDF generation...")
    chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
    
    # Absolute paths are required for Chrome headless to open files correctly on Windows
    abs_html_path = os.path.abspath(html_path)
    abs_pdf_path = os.path.join(base_dir, "whitepaper.pdf")
    
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf=" + abs_pdf_path,
        "--no-pdf-header-footer",
        "file:///" + abs_html_path.replace("\\", "/")
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"SUCCESS! Beautiful publication-grade PDF compiled successfully at: {abs_pdf_path}")
    except Exception as e:
        print(f"Error executing Chrome PDF render: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build()
