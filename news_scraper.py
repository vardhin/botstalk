import feedparser
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, List
import time
import re
from urllib.parse import urljoin

class RegionalNewsRSSCollector:
    """
    RSS-based news collector for regional I-LoRA training
    Much better than Selenium/BeautifulSoup for news!
    """
    
    def __init__(self):
        # Regional RSS feeds - major news sources
        self.regional_feeds = {
            "north_america": {
                "cnn": "http://rss.cnn.com/rss/edition.rss",
                "npr": "https://feeds.npr.org/1001/rss.xml",
                "cbc": "https://www.cbc.ca/cmlink/rss-topstories",
                "reuters_us": "https://feeds.reuters.com/reuters/domesticNews",
                "ap_news": "https://feeds.apnews.com/ApNews/apf-usnews",
                "fox_news": "http://feeds.foxnews.com/foxnews/latest",
                "wsj": "https://feeds.a.dj.com/rss/RSSWorldNews.xml"
            },
            
            "europe": {
                "bbc": "http://feeds.bbci.co.uk/news/world/europe/rss.xml",
                "reuters_eu": "https://feeds.reuters.com/reuters/europeanNews",
                "euronews": "https://feeds.feedburner.com/euronews/en/home",
                "dw": "https://rss.dw.com/rdf/rss-en-all",
                "france24": "https://www.france24.com/en/rss",
                "guardian": "https://www.theguardian.com/world/europe/rss"
            },
            
            "asia_pacific": {
                "nhk": "https://www3.nhk.or.jp/rss/news/cat0.xml",
                "scmp": "https://www.scmp.com/rss/91/feed",
                "japan_times": "https://www.japantimes.co.jp/rss/news/",
                "reuters_asia": "https://feeds.reuters.com/reuters/asianews",
                "abc_aus": "https://www.abc.net.au/news/feed/45910/rss.xml"
            },
            
            "middle_east": {
                "al_jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
                "reuters_me": "https://feeds.reuters.com/reuters/middleEastNews",
                "haaretz": "https://www.haaretz.com/cmlink/1.628455"
            },
            
            "africa": {
                "reuters_africa": "https://feeds.reuters.com/reuters/africanews",
                "bbc_africa": "http://feeds.bbci.co.uk/news/world/africa/rss.xml"
            },
            
            "latin_america": {
                "reuters_latam": "https://feeds.reuters.com/reuters/latinAmericanNews",
                "bbc_latam": "http://feeds.bbci.co.uk/news/world/latin_america/rss.xml"
            }
        }
    
    def collect_regional_articles(self, region: str, max_articles_per_feed: int = 50, 
                                days_back: int = 7) -> List[Dict]:
        """
        Collect articles from RSS feeds for a specific region
        
        Why RSS is better:
        1. No rate limiting issues
        2. Structured data (title, summary, link, date)
        3. Real-time updates
        4. No anti-bot measures
        5. Much faster than scraping
        """
        
        if region not in self.regional_feeds:
            print(f"Region {region} not found!")
            return []
        
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        print(f"Collecting articles for {region} from {len(self.regional_feeds[region])} RSS feeds...")
        
        for source_name, feed_url in self.regional_feeds[region].items():
            try:
                print(f"  Processing {source_name}...")
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                if feed.bozo:
                    print(f"    Warning: Feed parsing issues for {source_name}")
                
                feed_articles = 0
                for entry in feed.entries:
                    # Check date
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        else:
                            pub_date = datetime.now()  # Fallback
                        
                        if pub_date < cutoff_date:
                            continue
                    except:
                        pub_date = datetime.now()
                    
                    # Extract article data
                    article = {
                        "title": getattr(entry, 'title', '').strip(),
                        "summary": getattr(entry, 'summary', '').strip(),
                        "link": getattr(entry, 'link', ''),
                        "published": pub_date.isoformat(),
                        "source": source_name,
                        "region": region,
                        "content": ""  # Will be filled if needed
                    }
                    
                    # Clean summary (remove HTML tags)
                    article["summary"] = re.sub(r'<[^>]+>', '', article["summary"])
                    
                    # Quality filters
                    if (len(article["title"]) > 10 and 
                        len(article["summary"]) > 50 and
                        article["link"]):
                        
                        articles.append(article)
                        feed_articles += 1
                        
                        if feed_articles >= max_articles_per_feed:
                            break
                
                print(f"    Collected {feed_articles} articles from {source_name}")
                
                # Be nice to servers
                time.sleep(1)
                
            except Exception as e:
                print(f"    Error processing {source_name}: {e}")
                continue
        
        print(f"Total articles collected for {region}: {len(articles)}")
        return articles
    
    def enhance_articles_with_content(self, articles: List[Dict], 
                                    max_content_length: int = 2000) -> List[Dict]:
        """
        Optionally fetch full article content (use sparingly!)
        RSS summaries are usually sufficient for I-LoRA training
        """
        
        enhanced_articles = []
        
        for article in articles:
            try:
                # Try to get full content
                response = requests.get(article["link"], timeout=10)
                response.raise_for_status()
                
                # Basic content extraction (you could use newspaper3k here)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                # Extract main content (this is basic - could be improved)
                content = soup.get_text(separator=' ', strip=True)
                content = re.sub(r'\s+', ' ', content)[:max_content_length]
                
                article["content"] = content
                enhanced_articles.append(article)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"Could not fetch content for {article['title'][:50]}...: {e}")
                # Keep original article with just summary
                enhanced_articles.append(article)
        
        return enhanced_articles
    
    def convert_to_training_format(self, articles: List[Dict]) -> List[Dict]:
        """
        Convert news articles to I-LoRA training format
        """
        
        training_data = []
        
        for article in articles:
            # Use summary as primary content (RSS summaries are high quality)
            content = article["summary"]
            if article.get("content"):
                content = article["content"][:1000]  # Limit length
            
            # Create question-answer pairs from news articles
            training_examples = [
                {
                    "text": f"What happened regarding {article['title'][:100]}?",
                    "label": content,
                    "source": article["source"],
                    "region": article["region"],
                    "date": article["published"]
                },
                {
                    "text": f"Tell me about the news from {article['source']}",
                    "label": f"According to {article['source']}: {content}",
                    "source": article["source"],
                    "region": article["region"],
                    "date": article["published"]
                }
            ]
            
            training_data.extend(training_examples)
        
        return training_data


def integrate_with_ilora():
    """
    Integration with your I-LoRA trainer
    """
    
    # Collect regional news data
    collector = RegionalNewsRSSCollector()
    
    # Collect data for all regions
    all_regional_data = {}
    
    for region in collector.regional_feeds.keys():
        print(f"\n=== Collecting data for {region} ===")
        
        # Get articles via RSS
        articles = collector.collect_regional_articles(
            region, 
            max_articles_per_feed=30,  # 30 articles per feed
            days_back=3  # Last 3 days
        )
        
        # Convert to training format
        training_data = collector.convert_to_training_format(articles)
        
        # Store regional data
        all_regional_data[region] = training_data
        
        print(f"Region {region}: {len(articles)} articles → {len(training_data)} training examples")
        
        # Save regional data
        with open(f"regional_data_{region}.json", "w") as f:
            json.dump(training_data, f, indent=2)
    
    return all_regional_data


def run_news_enhanced_experiment():
    """
    Your I-LoRA experiment enhanced with real news data
    """
    
    # Collect real news data
    regional_data = integrate_with_ilora()
    
    # Import your I-LoRA trainer
    from main import ILoRATrainer
    
    # Initialize trainer
    trainer = ILoRATrainer(
        model_name="meta-llama/Llama-3.2-1B",
        lora_rank=16,
        lambda_interpolation=0.1
    )
    
    all_qa_pairs = []
    
    # Phase 1: Train on real regional news data
    print("\n=== Phase 1: Training on Real Regional News Data ===")
    
    for region, data in regional_data.items():
        if len(data) > 10:  # Only train if we have enough data
            print(f"\nTraining model for {region} with {len(data)} news examples...")
            
            # Train on regional news data
            trainer.train_regional_model(data[:100], region, epochs=2, batch_size=1)
            
            # Generate Q&A pairs
            regional_prompt = f"current news and events in {region.replace('_', ' ')}"
            qa_pairs = trainer.generate_qa_pairs(regional_prompt, num_questions=30)
            
            all_qa_pairs.extend(qa_pairs)
            
            print(f"Generated {len(qa_pairs)} Q&A pairs for {region}")
    
    # Phase 2: Consolidation
    print("\n=== Phase 2: Final Consolidation ===")
    if all_qa_pairs:
        trainer.train_regional_model(all_qa_pairs, "news_consolidated", epochs=3, batch_size=1)
    
    print(f"\nExperiment complete! Trained on real news data from {len(regional_data)} regions")


if __name__ == "__main__":
    # Run with real news data
    run_news_enhanced_experiment()