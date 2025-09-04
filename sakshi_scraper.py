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

class SakshiScraper:
    def __init__(self):
        self.base_url = "https://www.sakshi.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
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

    def extract_article_links(self, soup, base_url):
        """Extract article links from the main page"""
        article_links = []
        
        # Find all articles with class "news_link"
        article_elements = soup.find_all('a', class_='news_link')
        
        for element in article_elements:
            if element.get('href'):
                href = element.get('href')
                
                # Get title from the link text or from nested elements
                title = element.get_text(strip=True)
                if not title:
                    # Try to get title from nested h4 or other elements
                    title_elem = element.find(['h4', 'h3', 'h2', 'span'])
                    title = title_elem.get_text(strip=True) if title_elem else "No Title"
                
                # Convert relative URLs to absolute URLs
                if href.startswith('/'):
                    full_url = urljoin(base_url, href)
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = urljoin(base_url, href)
                
                # Avoid duplicate URLs
                if not any(article['url'] == full_url for article in article_links):
                    article_links.append({
                        'title': title,
                        'url': full_url
                    })
        
        return article_links

    def load_more_articles(self, url, max_pages=5):
        """Load more articles by checking for pagination or load more functionality"""
        all_article_links = []
        
        for page in range(1, max_pages + 1):
            # Try pagination URL patterns
            page_url = f"{url}?page={page}"
            
            logger.info(f"Fetching page {page}: {page_url}")
            html_content = self.get_page_content(page_url)
            
            if not html_content:
                logger.warning(f"Failed to fetch page {page}")
                break
                
            soup = BeautifulSoup(html_content, 'html.parser')
            page_links = self.extract_article_links(soup, self.base_url)
            
            if not page_links:
                logger.info(f"No articles found on page {page}, stopping pagination")
                break
                
            # Check for duplicate links (indicates we've reached the end)
            new_links = []
            for link in page_links:
                if not any(existing['url'] == link['url'] for existing in all_article_links):
                    new_links.append(link)
            
            if not new_links:
                logger.info("No new articles found, stopping pagination")
                break
                
            all_article_links.extend(new_links)
            logger.info(f"Found {len(new_links)} new articles on page {page}")
            
            # Add delay between page requests
            time.sleep(random.uniform(2, 4))
        
        return all_article_links
    
    def extract_article_content(self, article_url):
        """Extract article content from individual article page"""
        content_html = self.get_page_content(article_url)
        if not content_html:
            return None
        
        soup = BeautifulSoup(content_html, 'html.parser')
        
        # Extract title from h1 with class "news-heading"
        title_element = soup.find('h1', class_='news-heading')
        if not title_element:
            # Fallback to any h1 tag
            title_element = soup.find('h1')
        
        title = title_element.get_text(strip=True) if title_element else "No Title"
        
        # Extract content from div with class "news-story-content"
        content_div = soup.find('div', class_='news-story-content')
        if not content_div:
            logger.warning(f"No news-story-content div found in {article_url}")
            return None
        
        # Extract content from p tags within news-story-content
        content_paragraphs = []
        for p_tag in content_div.find_all('p'):
            text = p_tag.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short paragraphs
                content_paragraphs.append(text)
        
        # If no p tags found, get all text from the content div
        if not content_paragraphs:
            content = content_div.get_text(strip=True)
        else:
            content = '\n\n'.join(content_paragraphs)
        
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
            '.news-story-header .date',
            '.article-date',
            '.published-date',
            'time',
            '[datetime]'
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
        
        # Look for date patterns in the entire page
        try:
            # Look for JSON-LD structured data
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
    
    def scrape_articles(self, url, state_name, max_articles=2000, max_pages=100):
        """Main method to scrape articles from Sakshi website"""
        logger.info(f"Starting to scrape articles from {url} for {state_name}")
        
        # Load articles from multiple pages
        all_article_links = self.load_more_articles(url, max_pages)
        
        if not all_article_links:
            logger.error("No article links found")
            return []
        
        logger.info(f"Found {len(all_article_links)} total article links")
        
        scraped_articles = []
        
        for i, link_data in enumerate(all_article_links[:max_articles]):
            try:
                logger.info(f"Scraping article {i+1}/{min(len(all_article_links), max_articles)}: {link_data['title'][:50]}...")
                
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
    scraper = SakshiScraper()
    
    # Scrape articles from both states
    ap_articles = scraper.scrape_articles(
        url="https://www.sakshi.com/tags/andhra-pradesh",
        state_name="Andhra Pradesh",
        max_articles=1000,
        max_pages=50
    )
    
    ts_articles = scraper.scrape_articles(
        url="https://www.sakshi.com/tags/telangana", 
        state_name="Telangana",
        max_articles=1000,
        max_pages=50
    )
    
    # Print summary
    print(f"\nSakshi Scraping Summary:")
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