# 27-09-2025 (v0.0.3)
# Focused crawler: Follows <a> tags for structure and downloads <img> assets.
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# --- Configuration ---
MAIN_DOMAIN = "captainbarbershop.id"
BASE_URL = "https://captainbarbershop.id/"
OUTPUT_FOLDER = "captainbarbershop"

# A single set to track ALL processed URLs (pages, images, etc.) to prevent re-downloads
PROCESSED_URLS = set()

def create_local_path(url, output_folder):
    """
    Translates a URL into a local file path, creating an 'index.html'
    for clean URLs (e.g., '/about/').
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip('/')

    # If the path is empty (it's the root domain), create index.html
    if not path:
        path = 'index.html'
    # If the path has no file extension (e.g., '/about'), treat it as a directory
    # and create an index.html file inside it.
    elif not os.path.splitext(path)[1]:
        path = os.path.join(path, 'index.html')

    return os.path.join(output_folder, path)

def save_resource(url, local_filepath):
    """
    Fetches a single resource from a URL and saves it to the specified local path.
    Returns the content if it's HTML, otherwise returns None.
    """
    print(f"-> Processing: {url}")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"   - Failed (Status: {response.status_code})")
            return None

        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)

        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            # It's an HTML page
            soup = BeautifulSoup(response.text, 'html.parser')
            with open(local_filepath, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            print(f"   - Saved HTML to: {local_filepath}")
            return response.text
        else:
            # It's an image or other asset, save in binary mode
            with open(local_filepath, 'wb') as f:
                f.write(response.content)
            print(f"   - Saved Asset to: {local_filepath}")
            return None # Not HTML, so no content to parse

    except requests.exceptions.RequestException as e:
        print(f"   - Error: {e}")
        return None

def find_links_and_assets(html_content, base_url):
    """
    Parses HTML to find two types of resources:
    1. Pages to crawl (from <a> tags)
    2. Assets to download (from <img> tags)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    pages_to_crawl = set()
    assets_to_download = set()

    # 1. Find navigational links to new pages
    for tag in soup.find_all('a', href=True):
        href = tag['href']
        if href and not href.startswith(('mailto:', '#', 'tel:', 'javascript:')):
            full_url = urljoin(base_url, href)
            pages_to_crawl.add(full_url)

    # 2. Find image assets on the current page
    for tag in soup.find_all('img', src=True):
        src = tag['src']
        if src:
            full_url = urljoin(base_url, src)
            assets_to_download.add(full_url)
            
    return pages_to_crawl, assets_to_download

def crawl_page(url):
    """
    The main crawling function. Orchestrates fetching, saving, parsing, and recursion.
    """
    # 1. Check if we should process this URL
    if url in PROCESSED_URLS or MAIN_DOMAIN not in urlparse(url).netloc:
        return
    
    PROCESSED_URLS.add(url)
    
    # 2. Determine where to save the file and save it
    local_path = create_local_path(url, OUTPUT_FOLDER)
    html_content = save_resource(url, local_path)

    # 3. If it wasn't an HTML page, we stop here
    if not html_content:
        return

    # 4. If it was HTML, find all links and assets on it
    pages, assets = find_links_and_assets(html_content, url)
    
    # 5. Download the assets found on this page
    for asset_url in assets:
        if asset_url not in PROCESSED_URLS:
            PROCESSED_URLS.add(asset_url)
            asset_path = create_local_path(asset_url, OUTPUT_FOLDER)
            save_resource(asset_url, asset_path)

    # 6. Recursively crawl the new pages found
    for page_url in pages:
        crawl_page(page_url)


# --- Main execution ---
if __name__ == "__main__":
    print(f"🚀 Starting crawl of {BASE_URL}")
    print(f"   Output will be saved to '{OUTPUT_FOLDER}/'")
    crawl_page(BASE_URL)
    print("\n✅ Crawl finished.")