import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin, urlparse
import uuid
from datetime import datetime
import logging
import json
import re
from news_store import create_article

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EenaduScraper:
    def __init__(self):
        self.base_url = "https://www.eenadu.net"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'X-Requested-With': 'XMLHttpRequest',  # Important for AJAX requests
        })
        
    def get_page_content(self, url, retries=3):
        """Fetch page content with retry mechanism"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(1, 3))
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None

    def extract_load_more_params(self, soup):
        """Extract parameters needed for load more AJAX requests"""
        # Look for the load more button and extract necessary parameters
        load_more_button = soup.find('div', id='loadmore')
        if not load_more_button:
            logger.warning("Load more button not found")
            return None
        
        # Look for script tags that contain AJAX parameters
        scripts = soup.find_all('script')
        ajax_params = {}
        
        for script in scripts:
            if script.string:
                # Look for patterns that might contain AJAX parameters
                # Common patterns: offset, page, category_id, etc.
                
                # Try to find offset or page number
                offset_match = re.search(r'offset["\']?\s*[:=]\s*["\']?(\d+)', script.string, re.IGNORECASE)
                if offset_match:
                    ajax_params['offset'] = int(offset_match.group(1))
                
                page_match = re.search(r'page["\']?\s*[:=]\s*["\']?(\d+)', script.string, re.IGNORECASE)
                if page_match:
                    ajax_params['page'] = int(page_match.group(1))
                
                # Look for category or section ID
                cat_match = re.search(r'cat(?:egory)?[_]?id["\']?\s*[:=]\s*["\']?(\d+)', script.string, re.IGNORECASE)
                if cat_match:
                    ajax_params['cat_id'] = int(cat_match.group(1))
                
                # Look for AJAX URL
                ajax_url_match = re.search(r'["\']([^"\']*(?:ajax|load)[^"\']*)["\']', script.string, re.IGNORECASE)
                if ajax_url_match:
                    ajax_params['ajax_url'] = ajax_url_match.group(1)
        
        # Default values if not found
        if 'offset' not in ajax_params:
            ajax_params['offset'] = 10  # Common starting offset
        if 'page' not in ajax_params:
            ajax_params['page'] = 1
        
        return ajax_params

    def make_load_more_request(self, ajax_params, attempt_num):
        """Make AJAX request to load more articles"""
        # Common AJAX endpoints for load more functionality
        possible_endpoints = [
            f"{self.base_url}/ajax/loadmore.php",
            f"{self.base_url}/loadmore.php",
            f"{self.base_url}/ajax/load_articles.php",
            f"{self.base_url}/api/articles/load",
            f"{self.base_url}/andhra-pradesh/ajax/loadmore"
        ]
        
        # If we found a specific AJAX URL, try that first
        if 'ajax_url' in ajax_params:
            ajax_url = ajax_params['ajax_url']
            if not ajax_url.startswith('http'):
                ajax_url = urljoin(self.base_url, ajax_url)
            possible_endpoints.insert(0, ajax_url)
        
        # Prepare request data
        request_data = {
            'offset': ajax_params['offset'] + (attempt_num * 10),
            'page': ajax_params['page'] + attempt_num,
            'category': '255',  # Andhra Pradesh category ID (common)
            'limit': 10,
            'action': 'loadmore'
        }
        
        if 'cat_id' in ajax_params:
            request_data['cat_id'] = ajax_params['cat_id']
        
        for endpoint in possible_endpoints:
            try:
                logger.info(f"Trying AJAX endpoint: {endpoint}")
                
                # Try both POST and GET methods
                for method in ['POST', 'GET']:
                    try:
                        if method == 'POST':
                            response = self.session.post(endpoint, data=request_data, timeout=10)
                        else:
                            response = self.session.get(endpoint, params=request_data, timeout=10)
                        
                        if response.status_code == 200:
                            content = response.text.strip()
                            if content and len(content) > 100:  # Valid response
                                logger.info(f"Successfully loaded more content via {method} to {endpoint}")
                                return content
                    except Exception as e:
                        logger.debug(f"{method} request to {endpoint} failed: {e}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Failed to make request to {endpoint}: {e}")
                continue
        
        return None

    def perform_load_more_requests(self, initial_soup, max_attempts=20):
        """Perform multiple load more requests to get all available articles"""
        logger.info("Starting load more requests to expand article list")
        
        # Extract initial parameters
        ajax_params = self.extract_load_more_params(initial_soup)
        if not ajax_params:
            logger.warning("Could not extract AJAX parameters, proceeding with initial content only")
            return initial_soup
        
        all_html_content = [str(initial_soup)]
        successful_requests = 0
        
        for attempt in range(max_attempts):
            logger.info(f"Load more attempt {attempt + 1}/{max_attempts}")
            
            ajax_response = self.make_load_more_request(ajax_params, attempt)
            
            if ajax_response:
                # Try to parse as JSON first (common for AJAX responses)
                try:
                    json_data = json.loads(ajax_response)
                    if 'html' in json_data:
                        html_content = json_data['html']
                    elif 'content' in json_data:
                        html_content = json_data['content']
                    elif 'data' in json_data:
                        html_content = json_data['data']
                    else:
                        html_content = ajax_response
                except json.JSONDecodeError:
                    # Response is likely HTML
                    html_content = ajax_response
                
                # Check if we got valid content
                if html_content and len(html_content.strip()) > 50:
                    # Check if it contains article links
                    temp_soup = BeautifulSoup(html_content, 'html.parser')
                    article_elements = temp_soup.find_all('h3', class_='article-title-rgt')
                    
                    if article_elements:
                        all_html_content.append(html_content)
                        successful_requests += 1
                        logger.info(f"Load more successful, found {len(article_elements)} more articles")
                    else:
                        logger.info("No more articles found, stopping load more requests")
                        break
                else:
                    logger.info("Empty or invalid response, stopping load more requests")
                    break
            else:
                logger.info("Load more request failed, stopping")
                break
            
            # Add delay between requests
            time.sleep(random.uniform(1, 2))
        
        logger.info(f"Completed load more requests. Successful requests: {successful_requests}")
        
        # Combine all HTML content
        combined_html = '\n'.join(all_html_content)
        return BeautifulSoup(combined_html, 'html.parser')

    def extract_article_links(self, soup):
        """Extract article links from the main page"""
        article_links = []
        
        # Find all articles with class "article-title-rgt"
        article_elements = soup.find_all('h3', class_='article-title-rgt')
        
        for element in article_elements:
            link_tag = element.find('a') if element.name != 'a' else element
            if not link_tag:
                # Sometimes the h3 itself might be inside an a tag
                link_tag = element.find_parent('a')
            
            if link_tag and link_tag.get('href'):
                href = link_tag.get('href')
                title = link_tag.get_text(strip=True)
                
                # Convert relative URLs to absolute URLs
                if href.startswith('/'):
                    full_url = urljoin(self.base_url, href)
                else:
                    full_url = href
                
                # Avoid duplicate URLs
                if not any(article['url'] == full_url for article in article_links):
                    article_links.append({
                        'title': title,
                        'url': full_url
                    })
        
        return article_links
    
    def extract_article_content(self, article_url):
        """Extract article content from individual article page"""
        content_html = self.get_page_content(article_url)
        if not content_html:
            return None
        
        soup = BeautifulSoup(content_html, 'html.parser')
        
        # Find the fullstory div
        fullstory_div = soup.find('div', class_='fullstory')
        if not fullstory_div:
            logger.warning(f"No fullstory div found in {article_url}")
            return None
        
        # Extract content from p tags within fullstory
        content_paragraphs = []
        for p_tag in fullstory_div.find_all('p'):
            text = p_tag.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short paragraphs
                content_paragraphs.append(text)
        
        content = '\n\n'.join(content_paragraphs)
        
        # Extract title from h1 tag
        title_element = soup.find('h1', class_='red')
        if not title_element:
            title_element = soup.find('h1')
        
        title = title_element.get_text(strip=True) if title_element else "No Title"
        
        # Extract date
        date = self.extract_date(soup)
        
        return {
            'title': title,
            'content': content,
            'date': date,
            'url': article_url
        }
    
    def extract_date(self, soup):
        """Extract publication date from article page"""
        # Look for date in various possible locations
        date_selectors = [
            '.pub-sec span',
            '.pub-b span',
            'span:contains("Updated")',
            'span:contains("Published")'
        ]
        
        for selector in date_selectors:
            try:
                date_element = soup.select_one(selector)
                if date_element:
                    date_text = date_element.get_text(strip=True)
                    if any(keyword in date_text.lower() for keyword in ['updated', 'published', '2025', '2024']):
                        return date_text
            except:
                continue
        
        # Default to current date if not found
        return datetime.now().strftime("%Y-%m-%d")
    
    def scrape_articles(self, url="https://www.eenadu.net/andhra-pradesh", max_articles=2000):
        """Main method to scrape articles from Eenadu website"""
        logger.info(f"Starting to scrape articles from {url}")
        
        # Get initial page content
        html_content = self.get_page_content(url)
        if not html_content:
            logger.error("Failed to fetch main page")
            return []
        
        initial_soup = BeautifulSoup(html_content, 'html.parser')
        
        # Perform load more requests to get all available articles
        expanded_soup = self.perform_load_more_requests(initial_soup)
        
        # Extract article links from expanded content
        article_links = self.extract_article_links(expanded_soup)
        logger.info(f"Found {len(article_links)} total article links after load more expansion")
        
        scraped_articles = []
        
        for i, link_data in enumerate(article_links[:max_articles]):
            try:
                logger.info(f"Scraping article {i+1}/{min(len(article_links), max_articles)}: {link_data['title'][:50]}...")
                
                article_data = self.extract_article_content(link_data['url'])
                
                if article_data and article_data['content']:
                    # Generate unique ID
                    article_id = str(uuid.uuid4())
                    
                    # Prepare article data
                    final_article = {
                        'uid': article_id,
                        'title': article_data['title'],
                        'content': article_data['content'],
                        'date': article_data['date'],
                        'state': 'Andhra Pradesh',
                        'source_url': article_data['url']
                    }
                    
                    # Try to save to database
                    try:
                        success = create_article(
                            uid=final_article['uid'],
                            title=final_article['title'],
                            content=final_article['content'],
                            date=final_article['date'],
                            state=final_article['state']
                        )
                        
                        if success:
                            logger.info(f"Successfully saved article: {final_article['title'][:50]}...")
                            scraped_articles.append(final_article)
                        else:
                            logger.warning(f"Article already exists or failed to save: {final_article['title'][:50]}...")
                    
                    except Exception as e:
                        logger.error(f"Error saving article to database: {e}")
                        scraped_articles.append(final_article)  # Still add to return list
                
                # Add random delay between requests
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.error(f"Error scraping article {link_data['url']}: {e}")
                continue
        
        logger.info(f"Completed scraping. Total articles processed: {len(scraped_articles)}")
        return scraped_articles

def main():
    """Main function to run the scraper"""
    scraper = EenaduScraper()
    
    # Scrape articles
    articles = scraper.scrape_articles(max_articles=100)
    articles2 = scraper.scrape_articles(url="https://www.eenadu.net/telangana", max_articles=1000)
    # Print summary
    print(f"\nScraping Summary:")
    print(f"Total articles scraped: {len(articles)}")
    print(f"Total articles scraped from Telangana: {len(articles2)}")

    if articles:
        print(f"\nFirst few articles:")
        for i, article in enumerate(articles[:3]):
            print(f"\n{i+1}. Title: {article['title']}")
            print(f"   Date: {article['date']}")
            print(f"   Content preview: {article['content'][:100]}...")

    if articles2:
        print(f"\nFirst few articles from Telangana:")
        for i, article in enumerate(articles2[:3]):
            print(f"\n{i+1}. Title: {article['title']}")
            print(f"   Date: {article['date']}")
            print(f"   Content preview: {article['content'][:100]}...")

if __name__ == "__main__":
    main()