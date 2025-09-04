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

class TV9TeluguScraper:
    def __init__(self):
        self.base_url = "https://tv9telugu.com"
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
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(2, 5))
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None

    def extract_load_more_params(self, soup, page_url):
        """Extract parameters needed for load more AJAX requests"""
        # Look for the load more button
        load_more_button = soup.find(class_='load-more-btn')
        if not load_more_button:
            logger.warning("Load more button not found")
            return None
        
        # Look for script tags that contain AJAX parameters
        scripts = soup.find_all('script')
        ajax_params = {}
        
        for script in scripts:
            if script.string:
                # Look for patterns that might contain AJAX parameters
                
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
                
                # Look for section name
                section_match = re.search(r'section["\']?\s*[:=]\s*["\']([^"\']+)', script.string, re.IGNORECASE)
                if section_match:
                    ajax_params['section'] = section_match.group(1)
                
                # Look for AJAX URL
                ajax_url_match = re.search(r'["\']([^"\']*(?:ajax|load|more)[^"\']*)["\']', script.string, re.IGNORECASE)
                if ajax_url_match:
                    ajax_params['ajax_url'] = ajax_url_match.group(1)
        
        # Extract section from URL
        url_path = urlparse(page_url).path
        if url_path.startswith('/'):
            url_path = url_path[1:]
        if url_path:
            ajax_params['section'] = url_path
        
        # Default values if not found
        if 'offset' not in ajax_params:
            ajax_params['offset'] = 12  # Common starting offset for TV9
        if 'page' not in ajax_params:
            ajax_params['page'] = 1
        
        return ajax_params

    def make_load_more_request(self, ajax_params, attempt_num, page_url):
        """Make AJAX request to load more articles"""
        # Common AJAX endpoints for load more functionality
        possible_endpoints = [
            f"{self.base_url}/wp-admin/admin-ajax.php",
            f"{self.base_url}/ajax/loadmore.php",
            f"{self.base_url}/loadmore",
            f"{self.base_url}/api/load-more",
            f"{self.base_url}/wp-json/tv9/v1/posts"
        ]
        
        # If we found a specific AJAX URL, try that first
        if 'ajax_url' in ajax_params:
            ajax_url = ajax_params['ajax_url']
            if not ajax_url.startswith('http'):
                ajax_url = urljoin(self.base_url, ajax_url)
            possible_endpoints.insert(0, ajax_url)
        
        # Prepare request data for WordPress-style AJAX
        request_data = {
            'action': 'load_more_posts',
            'offset': ajax_params['offset'] + (attempt_num * 12),
            'page': ajax_params['page'] + attempt_num,
            'posts_per_page': 12,
            'category': ajax_params.get('section', ''),
            'section': ajax_params.get('section', ''),
            'nonce': '',  # TV9 might use nonce for security
        }
        
        if 'cat_id' in ajax_params:
            request_data['cat_id'] = ajax_params['cat_id']
        
        # Also try with simpler parameters
        simple_data = {
            'offset': request_data['offset'],
            'limit': 12,
            'section': ajax_params.get('section', ''),
        }
        
        for endpoint in possible_endpoints:
            try:
                logger.info(f"Trying AJAX endpoint: {endpoint}")
                
                # Try both POST and GET methods with different data formats
                for method, data in [('POST', request_data), ('POST', simple_data), ('GET', simple_data)]:
                    try:
                        if method == 'POST':
                            response = self.session.post(endpoint, data=data, timeout=15)
                        else:
                            response = self.session.get(endpoint, params=data, timeout=15)
                        
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

    def perform_load_more_requests(self, initial_soup, page_url, max_attempts=20):
        """Perform multiple load more requests to get all available articles"""
        logger.info("Starting load more requests to expand article list")
        
        # Extract initial parameters
        ajax_params = self.extract_load_more_params(initial_soup, page_url)
        if not ajax_params:
            logger.warning("Could not extract AJAX parameters, proceeding with initial content only")
            return initial_soup
        
        all_html_content = [str(initial_soup)]
        successful_requests = 0
        
        for attempt in range(max_attempts):
            logger.info(f"Load more attempt {attempt + 1}/{max_attempts}")
            
            ajax_response = self.make_load_more_request(ajax_params, attempt, page_url)
            
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
                    elif 'posts' in json_data:
                        html_content = json_data['posts']
                    else:
                        html_content = ajax_response
                except json.JSONDecodeError:
                    # Response is likely HTML
                    html_content = ajax_response
                
                # Check if we got valid content
                if html_content and len(html_content.strip()) > 50:
                    # Check if it contains article links
                    temp_soup = BeautifulSoup(html_content, 'html.parser')
                    figure_elements = temp_soup.find_all('figure')
                    
                    if figure_elements:
                        all_html_content.append(html_content)
                        successful_requests += 1
                        logger.info(f"Load more successful, found {len(figure_elements)} more articles")
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
            time.sleep(random.uniform(2, 4))
        
        logger.info(f"Completed load more requests. Successful requests: {successful_requests}")
        
        # Combine all HTML content
        combined_html = '\n'.join(all_html_content)
        return BeautifulSoup(combined_html, 'html.parser')

    def extract_article_links(self, soup):
        """Extract article links from figure tags"""
        article_links = []
        
        # Find all figure tags that contain article links
        figure_elements = soup.find_all('figure')
        
        for figure in figure_elements:
            # Look for anchor tags within the figure
            link_tag = figure.find('a')
            if link_tag and link_tag.get('href'):
                href = link_tag.get('href')
                
                # Get title from various possible sources
                title = ""
                
                # Try to get title from img alt text
                img_tag = link_tag.find('img')
                if img_tag and img_tag.get('alt'):
                    title = img_tag.get('alt')
                
                # Try to get title from the link text
                if not title:
                    title = link_tag.get_text(strip=True)
                
                # Try to get title from data attributes
                if not title:
                    title = link_tag.get('title', '')
                
                # Try to find title in sibling elements
                if not title:
                    sibling_h3 = figure.find_next_sibling(['h3', 'h2', 'h4'])
                    if sibling_h3:
                        sibling_link = sibling_h3.find('a')
                        if sibling_link:
                            title = sibling_link.get_text(strip=True)
                
                # Convert relative URLs to absolute URLs
                if href.startswith('/'):
                    full_url = urljoin(self.base_url, href)
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = urljoin(self.base_url, href)
                
                # Avoid duplicate URLs and ensure it's a valid article URL
                if (not any(article['url'] == full_url for article in article_links) and 
                    ('tv9telugu.com' in full_url) and 
                    title):
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
        
        # Extract title from element with class "article-HD"
        title_element = soup.find(class_='article-HD')
        if not title_element:
            # Fallback to h1 tags
            title_element = soup.find('h1')
        
        title = title_element.get_text(strip=True) if title_element else "No Title"
        
        # Extract short description
        short_desc_element = soup.find(class_='short_desc')
        short_desc = short_desc_element.get_text(strip=True) if short_desc_element else ""
        
        # Extract main content from ArticleBodyCont
        content_div = soup.find(class_='ArticleBodyCont')
        content_paragraphs = []
        
        if content_div:
            # Extract content from p tags within ArticleBodyCont
            for p_tag in content_div.find_all('p'):
                text = p_tag.get_text(strip=True)
                if text and len(text) > 10:  # Filter out very short paragraphs
                    content_paragraphs.append(text)
        else:
            logger.warning(f"No ArticleBodyCont div found in {article_url}")
        
        # Combine short description and main content
        all_content = []
        if short_desc:
            all_content.append(short_desc)
        if content_paragraphs:
            all_content.extend(content_paragraphs)
        
        content = '\n\n'.join(all_content)
        
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
            '.article-date',
            '.published-date',
            '.post-date',
            'time',
            '[datetime]',
            '.date',
            '.timestamp'
        ]
        
        for selector in date_selectors:
            try:
                date_element = soup.select_one(selector)
                if date_element:
                    # Try to get datetime attribute first
                    date_text = date_element.get('datetime')
                    if not date_text:
                        date_text = date_element.get_text(strip=True)
                    
                    if date_text and any(keyword in date_text.lower() for keyword in ['2025', '2024', '2023', 'updated', 'published']):
                        return date_text
            except:
                continue
        
        # Look for date patterns in JSON-LD structured data
        try:
            json_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        date_published = data.get('datePublished')
                        if date_published:
                            return date_published
                        date_modified = data.get('dateModified')
                        if date_modified:
                            return date_modified
                except:
                    continue
        except:
            pass
        
        # Default to current date if not found
        return datetime.now().strftime("%Y-%m-%d")
    
    def scrape_articles(self, url, state_name, max_articles=2000):
        """Main method to scrape articles from TV9 Telugu website"""
        logger.info(f"Starting to scrape articles from {url} for {state_name}")
        
        # Get initial page content
        html_content = self.get_page_content(url)
        if not html_content:
            logger.error("Failed to fetch main page")
            return []
        
        initial_soup = BeautifulSoup(html_content, 'html.parser')
        
        # Perform load more requests to get all available articles
        expanded_soup = self.perform_load_more_requests(initial_soup, url)
        
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
                        'state': state_name,
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
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                logger.error(f"Error scraping article {link_data['url']}: {e}")
                continue
        
        logger.info(f"Completed scraping {state_name}. Total articles processed: {len(scraped_articles)}")
        return scraped_articles

def main():
    """Main function to run the scraper"""
    scraper = TV9TeluguScraper()
    
    # Scrape articles from both states
    ap_articles = scraper.scrape_articles(
        url="https://tv9telugu.com/andhra-pradesh",
        state_name="Andhra Pradesh",
        max_articles=1000
    )
    
    ts_articles = scraper.scrape_articles(
        url="https://tv9telugu.com/telangana", 
        state_name="Telangana",
        max_articles=1000
    )
    
    # Print summary
    print(f"\nTV9 Telugu Scraping Summary:")
    print(f"Total articles scraped from Andhra Pradesh: {len(ap_articles)}")
    print(f"Total articles scraped from Telangana: {len(ts_articles)}")
    print(f"Total articles: {len(ap_articles) + len(ts_articles)}")

    if ap_articles:
        print(f"\nFirst few Andhra Pradesh articles:")
        for i, article in enumerate(ap_articles[:3]):
            print(f"\n{i+1}. Title: {article['title']}")
            print(f"   Date: {article['date']}")
            print(f"   Content preview: {article['content'][:100]}...")

    if ts_articles:
        print(f"\nFirst few Telangana articles:")
        for i, article in enumerate(ts_articles[:3]):
            print(f"\n{i+1}. Title: {article['title']}")
            print(f"   Date: {article['date']}")
            print(f"   Content preview: {article['content'][:100]}...")

if __name__ == "__main__":
    main()