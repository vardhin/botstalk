import click
import sqlite3
from datetime import datetime
from typing import Dict, Any
import uuid
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import track
from rich import box
from rich.layout import Layout
from rich.columns import Columns
import textwrap

# Import news store functions
from news_store import (
    create_article, get_article_by_uid, get_articles_by_state,
    get_articles_by_date, get_articles_in_date_range, update_article,
    delete_article, fuzzy_search, DB_PATH
)

console = Console()

class NewsStoreCLI:
    def __init__(self):
        self.console = console
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total articles
        c.execute("SELECT COUNT(*) FROM articles")
        total_articles = c.fetchone()[0]
        
        # Articles by state
        c.execute("SELECT state, COUNT(*) FROM articles GROUP BY state")
        states = dict(c.fetchall())
        
        # Total word count
        c.execute("SELECT content FROM articles")
        contents = c.fetchall()
        total_words = sum(len(content[0].split()) for content in contents if content[0])
        
        # Average words per article
        avg_words = total_words / total_articles if total_articles > 0 else 0
        
        # Recent articles (last 7 days)
        c.execute("SELECT COUNT(*) FROM articles WHERE date >= date('now', '-7 days')")
        recent_articles = c.fetchone()[0]
        
        conn.close()
        
        return {
            'total_articles': total_articles,
            'states': states,
            'total_words': total_words,
            'avg_words': round(avg_words, 1),
            'recent_articles': recent_articles
        }
    
    def display_banner(self):
        """Display welcome banner"""
        banner = Text.assemble(
            ("📰 ", "bold blue"),
            ("NewsStore CLI", "bold white"),
            (" v1.0", "dim white")
        )
        panel = Panel(
            banner,
            box=box.ROUNDED,
            style="blue",
            width=50
        )
        self.console.print(panel, justify="center")
    
    def display_stats(self):
        """Display database statistics"""
        stats = self.get_stats()
        
        # Create stats panels
        panels = []
        
        # Main stats
        main_stats = f"""
📊 Total Articles: [bold cyan]{stats['total_articles']}[/bold cyan]
📝 Total Words: [bold green]{stats['total_words']:,}[/bold green]
📈 Avg Words/Article: [bold yellow]{stats['avg_words']}[/bold yellow]
🔥 Recent (7 days): [bold red]{stats['recent_articles']}[/bold red]
        """.strip()
        
        panels.append(Panel(main_stats, title="📊 Statistics", box=box.ROUNDED))
        
        # State distribution
        if stats['states']:
            state_info = "\n".join([
                f"[cyan]{state}[/cyan]: {count}" 
                for state, count in stats['states'].items()
            ])
        else:
            state_info = "[dim]No articles found[/dim]"
            
        panels.append(Panel(state_info, title="📋 By State", box=box.ROUNDED))
        
        self.console.print(Columns(panels, equal=True))
    
    def display_article(self, article):
        """Display a single article in a nice format"""
        if not article:
            self.console.print("[red]Article not found[/red]")
            return
            
        uid, title, content, date, state = article
        
        # Truncate content for display
        display_content = textwrap.fill(content[:200] + "..." if len(content) > 200 else content, width=70)
        
        article_info = f"""
[bold cyan]UID:[/bold cyan] {uid}
[bold green]Date:[/bold green] {date}
[bold yellow]State:[/bold yellow] {state}

[bold white]Content:[/bold white]
{display_content}
        """.strip()
        
        panel = Panel(
            article_info,
            title=f"📰 {title}",
            box=box.ROUNDED,
            style="white"
        )
        self.console.print(panel)
    
    def display_articles_table(self, articles, title="Articles"):
        """Display articles in a table format"""
        if not articles:
            self.console.print(f"[yellow]No articles found[/yellow]")
            return
            
        table = Table(title=f"📰 {title}", box=box.ROUNDED)
        table.add_column("UID", style="cyan", width=8)
        table.add_column("Title", style="white", width=25)
        table.add_column("Date", style="green", width=12)
        table.add_column("State", style="yellow", width=10)
        table.add_column("Words", style="magenta", width=8)
        
        for article in articles:
            uid, title, content, date, state = article
            word_count = len(content.split()) if content else 0
            # Truncate title if too long
            display_title = title[:22] + "..." if len(title) > 25 else title
            table.add_row(
                uid[:8],
                display_title,
                date,
                state,
                str(word_count)
            )
        
        self.console.print(table)

cli = NewsStoreCLI()

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """📰 NewsStore CLI - Manage your news articles with style"""
    if ctx.invoked_subcommand is None:
        cli.display_banner()
        console.print()
        cli.display_stats()
        console.print()
        console.print("[dim]Use --help to see available commands[/dim]")

@main.command()
def stats():
    """📊 Show database statistics"""
    cli.display_stats()

@main.command()
@click.option('--title', prompt='Article title', help='Title of the article')
@click.option('--content', prompt='Article content', help='Content of the article')
@click.option('--state', prompt='Article state', help='State of the article')
@click.option('--date', help='Date (YYYY-MM-DD, defaults to today)')
def add(title, content, state, date):
    """➕ Add a new article"""
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    uid = str(uuid.uuid4())[:8]  # Short UID for display
    
    with console.status("[bold green]Creating article..."):
        success = create_article(uid, title, content, date, state)
    
    if success:
        console.print(f"[green]✅ Article created successfully![/green]")
        console.print(f"[cyan]UID: {uid}[/cyan]")
    else:
        console.print("[red]❌ Failed to create article (title might already exist)[/red]")

@main.command()
@click.argument('uid')
def get(uid):
    """🔍 Get article by UID"""
    article = get_article_by_uid(uid)
    cli.display_article(article)

@main.command()
@click.argument('state')
def state(state):
    """📋 Get articles by state"""
    articles = get_articles_by_state(state)
    cli.display_articles_table(articles, f"Articles in state: {state}")

@main.command()
@click.argument('date')
def date(date):
    """📅 Get articles by date (YYYY-MM-DD)"""
    articles = get_articles_by_date(date)
    cli.display_articles_table(articles, f"Articles from: {date}")

@main.command()
@click.argument('start_date')
@click.argument('end_date')
def range(start_date, end_date):
    """📊 Get articles in date range"""
    articles = get_articles_in_date_range(start_date, end_date)
    cli.display_articles_table(articles, f"Articles from {start_date} to {end_date}")

@main.command()
@click.argument('uid')
@click.option('--title', help='New title')
@click.option('--content', help='New content')
@click.option('--state', help='New state')
@click.option('--date', help='New date')
def update(uid, title, content, state, date):
    """✏️ Update an article"""
    # Check if article exists
    article = get_article_by_uid(uid)
    if not article:
        console.print("[red]❌ Article not found[/red]")
        return
    
    with console.status("[bold yellow]Updating article..."):
        update_article(uid, title, content, date, state)
    
    console.print("[green]✅ Article updated successfully![/green]")

@main.command()
@click.argument('uid')
def delete(uid):
    """🗑️ Delete an article"""
    # Check if article exists
    article = get_article_by_uid(uid)
    if not article:
        console.print("[red]❌ Article not found[/red]")
        return
    
    # Show article info
    cli.display_article(article)
    
    if Confirm.ask("\n[red]Are you sure you want to delete this article?[/red]"):
        with console.status("[bold red]Deleting article..."):
            delete_article(uid)
        console.print("[green]✅ Article deleted successfully![/green]")
    else:
        console.print("[yellow]Deletion cancelled[/yellow]")

@main.command()
@click.argument('query')
@click.option('--limit', '-n', default=10, help='Maximum number of results')
def search(query, limit):
    """🔍 Search articles (fuzzy search)"""
    with console.status(f"[bold blue]Searching for '{query}'..."):
        articles = fuzzy_search(query, limit)
    
    cli.display_articles_table(articles, f"Search results for: '{query}'")

@main.command()
def list():
    """📋 List all articles"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM articles ORDER BY date DESC")
    articles = c.fetchall()
    conn.close()
    
    cli.display_articles_table(articles, "All Articles")

@main.command()
def interactive():
    """🎯 Interactive mode"""
    console.print("[bold blue]🎯 Interactive NewsStore CLI[/bold blue]")
    console.print("[dim]Type 'exit' to quit[/dim]\n")
    
    while True:
        try:
            command = Prompt.ask("\n[cyan]newsstore[/cyan]", default="stats")
            
            if command.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            elif command == 'stats':
                cli.display_stats()
            elif command == 'list':
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT * FROM articles ORDER BY date DESC LIMIT 10")
                articles = c.fetchall()
                conn.close()
                cli.display_articles_table(articles, "Recent Articles (10)")
            elif command.startswith('search '):
                query = command[7:]
                articles = fuzzy_search(query, 5)
                cli.display_articles_table(articles, f"Search: '{query}'")
            elif command.startswith('get '):
                uid = command[4:]
                article = get_article_by_uid(uid)
                cli.display_article(article)
            else:
                console.print(f"[yellow]Unknown command: {command}[/yellow]")
                console.print("[dim]Available: stats, list, search <query>, get <uid>, exit[/dim]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break

if __name__ == '__main__':
    main()