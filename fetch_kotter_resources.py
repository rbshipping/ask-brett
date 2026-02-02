"""
Kotter Resources Fetcher
Fetches John Kotter change management resources from various URLs.

Run with: python fetch_kotter_resources.py

After running, process with:
    python process_documents.py --input kotter_knowledge --subfolder "Kotter Change Management"
    python build_index.py
"""

import os
import re
import time
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote

# HTML parsing
from bs4 import BeautifulSoup

# PDF extraction
import pdfplumber

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "kotter_knowledge"

# Request settings
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 2  # seconds

# URLs to fetch
URLS = [
    # Web pages
    "https://www.kotterinc.com/methodology/8-steps/",
    "https://www.kotterinc.com/methodology/the-4-core-change-principles-dr-john-kotter/",
    "https://www.kotterinc.com/methodology/",
    "https://www.kotterinc.com/research-insights/",
    "https://www.kotterinc.com/bringing-strategy-vision-to-life/",
    "https://www.kotterinc.com/leadership-in-an-era-of-constant-change-growing-bouncing-back-and-developing-with-purpose/",
    "https://www.kotterinc.com/emerging-thought-leadership/",
    "https://www.kotterinc.com/services/leadership-development/",
    "https://www.kotterinc.com/bookshelf/",
    "https://www.kotterinc.com/bookshelf/leading-change/",
    "https://www.kotterinc.com/bookshelf/the-heart-of-change/",
    "https://www.kotterinc.com/bookshelf/our-iceberg-is-melting-2/",
    "https://www.kotterinc.com/bookshelf/buy-in/",
    "https://www.kotterinc.com/bookshelf/change/",
    "https://www.hbs.edu/faculty/Pages/profile.aspx?facId=6495&view=publications",

    # PDFs
    "https://www.kotterinc.com/wp-content/uploads/2019/04/8-Steps-eBook-Kotter-2018.pdf",
    "https://www.kotterinc.com/wp-content/uploads/2017/06/OFFICIAL-_-Accelerate-HBR-Nov_2012_print-1.pdf",
    "https://www.kotterinc.com/wp-content/uploads/2020/10/Leading-Rapid-Change-Guide-Kotter.pdf",
    "https://irp-cdn.multiscreensite.com/6e5efd05/files/uploaded/Leading%20Change.pdf",
    "https://www.edomi.org/wp-content/uploads/2021/01/What-Leaders-Really-Do-kotter.pdf",

    # Special handling needed (may not work)
    "https://www.youtube.com/watch?v=gOhYT737ItY",
    "https://www.linkedin.com/in/johnkotter/",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_filename(name):
    """Create a clean filename from text"""
    # Remove invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace spaces and multiple dashes
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'-+', '-', name)
    name = re.sub(r'_+', '_', name)
    # Limit length
    return name[:80].strip('_-')


def clean_text(text):
    """Clean extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


def get_filename_from_url(url):
    """Generate a descriptive filename from URL"""
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # For PDFs, use the PDF filename
    if path.endswith('.pdf'):
        name = Path(path).stem
        return clean_filename(name) + '.txt'

    # For web pages, use the path
    path_parts = [p for p in path.strip('/').split('/') if p]
    if path_parts:
        name = '_'.join(path_parts[-2:]) if len(path_parts) > 1 else path_parts[-1]
        return clean_filename(f"kotter_{name}") + '.txt'

    # Fallback to domain
    return clean_filename(f"kotter_{parsed.netloc}") + '.txt'


# =============================================================================
# CONTENT EXTRACTORS
# =============================================================================

def extract_html_content(html, url):
    """Extract main text content from HTML"""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted elements
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
        tag.decompose()

    # Try to find main content area
    main_content = None
    for selector in ['main', 'article', '.content', '.post-content', '.entry-content', '#content', '.main-content']:
        if selector.startswith('.') or selector.startswith('#'):
            main_content = soup.select_one(selector)
        else:
            main_content = soup.find(selector)
        if main_content:
            break

    if not main_content:
        main_content = soup.find('body') or soup

    # Get title
    title = ""
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text().strip()

    h1 = soup.find('h1')
    if h1:
        title = h1.get_text().strip()

    # Extract text
    text_parts = []

    if title:
        text_parts.append(f"TITLE: {title}")
        text_parts.append(f"SOURCE: {url}")
        text_parts.append("=" * 60)
        text_parts.append("")

    # Process content elements
    for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'blockquote']):
        text = element.get_text().strip()
        if not text:
            continue

        if element.name.startswith('h'):
            level = int(element.name[1])
            prefix = '#' * level
            text_parts.append(f"\n{prefix} {text}\n")
        elif element.name == 'li':
            text_parts.append(f"• {text}")
        elif element.name == 'blockquote':
            text_parts.append(f'> "{text}"')
        else:
            text_parts.append(text)

    return clean_text('\n'.join(text_parts))


def extract_pdf_content(pdf_path, url):
    """Extract text from PDF"""
    text_parts = []
    text_parts.append(f"SOURCE: {url}")
    text_parts.append(f"FILE: {pdf_path.name}")
    text_parts.append("=" * 60)
    text_parts.append("")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"\n--- Page {i} ---\n")
                    text_parts.append(page_text)
    except Exception as e:
        text_parts.append(f"\n[Error extracting PDF: {e}]")

    return clean_text('\n'.join(text_parts))


# =============================================================================
# FETCHERS
# =============================================================================

def fetch_webpage(url):
    """Fetch and extract content from a webpage"""
    print(f"  Fetching webpage: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        content = extract_html_content(response.text, url)

        if len(content) < 200:
            print(f"    Warning: Very little content extracted ({len(content)} chars)")
            return None

        return content

    except requests.exceptions.RequestException as e:
        print(f"    Error fetching: {e}")
        return None


def fetch_pdf(url):
    """Download and extract content from a PDF"""
    print(f"  Fetching PDF: {url}")

    # Create temp directory for PDFs
    temp_dir = OUTPUT_DIR / "temp_pdfs"
    temp_dir.mkdir(exist_ok=True)

    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()

        # Save PDF temporarily
        pdf_name = Path(unquote(urlparse(url).path)).name
        pdf_path = temp_dir / pdf_name

        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        print(f"    Downloaded: {pdf_name}")

        # Extract text
        content = extract_pdf_content(pdf_path, url)

        # Clean up temp file
        pdf_path.unlink()

        if len(content) < 200:
            print(f"    Warning: Very little content extracted ({len(content)} chars)")
            return None

        return content

    except requests.exceptions.RequestException as e:
        print(f"    Error fetching: {e}")
        return None
    except Exception as e:
        print(f"    Error processing PDF: {e}")
        return None


def fetch_youtube(url):
    """Create a placeholder for YouTube content"""
    print(f"  YouTube URL: {url}")
    print(f"    Note: YouTube transcripts require manual extraction or API access")

    # Extract video ID
    video_id = None
    if 'v=' in url:
        video_id = url.split('v=')[1].split('&')[0]

    content = f"""TITLE: John Kotter YouTube Video
SOURCE: {url}
VIDEO_ID: {video_id}
{"=" * 60}

This is a YouTube video. To add the content:

1. Visit the URL: {url}
2. Enable captions/subtitles
3. Use YouTube's transcript feature (click "..." under the video, then "Show transcript")
4. Copy the transcript text and replace this placeholder

Alternatively, search for "John Kotter" transcripts or summaries online.

Topic: John Kotter on Change Management (based on URL context)
"""
    return content


def fetch_linkedin(url):
    """Create a placeholder for LinkedIn content"""
    print(f"  LinkedIn URL: {url}")
    print(f"    Note: LinkedIn requires authentication, creating placeholder")

    content = f"""TITLE: John Kotter LinkedIn Profile
SOURCE: {url}
{"=" * 60}

LinkedIn profiles require authentication to access.

To add this content:
1. Log into LinkedIn
2. Visit: {url}
3. Copy relevant sections (About, Experience, Articles, etc.)
4. Replace this placeholder with the copied content

Key information to capture:
- Professional summary/about section
- Recent posts about change management
- Published articles
- Key career highlights
"""
    return content


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_url(url):
    """Process a single URL and return (filename, content)"""
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Determine content type and fetch
    if path.endswith('.pdf'):
        content = fetch_pdf(url)
    elif 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
        content = fetch_youtube(url)
    elif 'linkedin.com' in parsed.netloc:
        content = fetch_linkedin(url)
    else:
        content = fetch_webpage(url)

    if not content:
        return None, None

    filename = get_filename_from_url(url)
    return filename, content


def main():
    print("\n" + "=" * 60)
    print("  KOTTER RESOURCES FETCHER")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Processing {len(URLS)} URLs...\n")

    # Process each URL
    success_count = 0
    failed_urls = []

    for i, url in enumerate(URLS, 1):
        print(f"\n[{i}/{len(URLS)}] Processing...")

        filename, content = process_url(url)

        if content:
            # Save content
            output_path = OUTPUT_DIR / filename

            # Handle duplicate filenames
            if output_path.exists():
                base = output_path.stem
                ext = output_path.suffix
                counter = 2
                while output_path.exists():
                    output_path = OUTPUT_DIR / f"{base}_{counter}{ext}"
                    counter += 1

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"    Saved: {output_path.name} ({len(content):,} chars)")
            success_count += 1
        else:
            failed_urls.append(url)
            print(f"    FAILED: Could not extract content")

        # Delay between requests
        if i < len(URLS):
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Clean up temp directory
    temp_dir = OUTPUT_DIR / "temp_pdfs"
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except:
            pass

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully processed: {success_count}/{len(URLS)}")
    print(f"Output directory: {OUTPUT_DIR}")

    if failed_urls:
        print(f"\nFailed URLs ({len(failed_urls)}):")
        for url in failed_urls:
            print(f"  - {url}")

    print("\n" + "-" * 60)
    print("NEXT STEPS:")
    print("-" * 60)
    print("1. Review the extracted content in the kotter_knowledge folder")
    print("2. Fill in YouTube/LinkedIn placeholders manually if needed")
    print("3. Process into chunks:")
    print('   python process_documents.py --input kotter_knowledge --subfolder "Kotter"')
    print("4. Rebuild the search index:")
    print("   python build_index.py")
    print()


if __name__ == "__main__":
    main()
