import click
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
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
from rich.align import Align
import textwrap
import time
from collections import defaultdict

# Import training data store functions
from training_data_store import (
    create_training_data, get_training_data_by_uuid, get_training_data_by_state,
    get_training_data_by_article_uuid, get_training_data_by_master_model,
    update_training_data, delete_training_data, fuzzy_search, DB_PATH
)

console = Console()

class TrainingDataCLI:
    def __init__(self):
        self.console = console
        self.current_index = 0
        self.training_data = []
        self.filter_state = None
        self.filter_model = None
        self.search_query = None
        self.running = True
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Total training data
        c.execute("SELECT COUNT(*) FROM training_data")
        total_data = c.fetchone()[0]
        
        # Data by state
        c.execute("SELECT state, COUNT(*) FROM training_data GROUP BY state")
        states = dict(c.fetchall())
        
        # Data by model
        c.execute("SELECT master_model, COUNT(*) FROM training_data GROUP BY master_model")
        models = dict(c.fetchall())
        
        # Articles with training data
        c.execute("SELECT COUNT(DISTINCT uuid_of_used_article) FROM training_data WHERE uuid_of_used_article IS NOT NULL")
        articles_with_data = c.fetchone()[0]
        
        # Average Q&A pairs per article
        c.execute("SELECT uuid_of_used_article, COUNT(*) FROM training_data WHERE uuid_of_used_article IS NOT NULL GROUP BY uuid_of_used_article")
        pairs_per_article = c.fetchall()
        avg_pairs = sum(count for _, count in pairs_per_article) / len(pairs_per_article) if pairs_per_article else 0
        
        # Total words in questions and answers (fixed calculation)
        c.execute("SELECT question, answer FROM training_data")
        qa_pairs = c.fetchall()
        total_words = 0
        valid_pairs = 0
        for question, answer in qa_pairs:
            if question and answer and str(question).strip() and str(answer).strip():
                total_words += len(str(question).split()) + len(str(answer).split())
                valid_pairs += 1
        
        # Average words per Q&A pair
        avg_words = total_words / valid_pairs if valid_pairs > 0 else 0
        
        # Recent training data (last 100 entries)
        c.execute("SELECT COUNT(*) FROM training_data WHERE rowid > (SELECT MAX(rowid) - 100 FROM training_data)")
        recent_data = c.fetchone()[0]
        
        # Additional detailed stats
        # Average question length
        c.execute("SELECT question FROM training_data WHERE question IS NOT NULL AND question != ''")
        questions = [row[0] for row in c.fetchall() if row[0] and str(row[0]).strip()]
        avg_question_length = sum(len(str(q).split()) for q in questions) / len(questions) if questions else 0
        
        # Average answer length
        c.execute("SELECT answer FROM training_data WHERE answer IS NOT NULL AND answer != ''")
        answers = [row[0] for row in c.fetchall() if row[0] and str(row[0]).strip()]
        avg_answer_length = sum(len(str(a).split()) for a in answers) / len(answers) if answers else 0
        
        # Character counts
        total_chars = sum(len(str(q)) + len(str(a)) for q, a in qa_pairs if q and a)
        
        conn.close()
        
        return {
            'total_data': total_data,
            'states': states,
            'models': models,
            'articles_with_data': articles_with_data,
            'avg_pairs': round(avg_pairs, 1),
            'total_words': total_words,
            'avg_words': round(avg_words, 1),
            'recent_data': recent_data,
            'valid_pairs': valid_pairs,
            'avg_question_length': round(avg_question_length, 1),
            'avg_answer_length': round(avg_answer_length, 1),
            'total_chars': total_chars
        }

    def load_training_data(self):
        """Load training data based on current filters"""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        query = "SELECT * FROM training_data"
        params = []
        conditions = []
        
        if self.filter_state:
            conditions.append("state = ?")
            params.append(self.filter_state)
            
        if self.filter_model:
            conditions.append("master_model = ?")
            params.append(self.filter_model)
            
        if self.search_query:
            conditions.append("(question LIKE ? OR answer LIKE ?)")
            params.extend([f"%{self.search_query}%", f"%{self.search_query}%"])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY rowid DESC"
        
        c.execute(query, params)
        self.training_data = c.fetchall()
        conn.close()
        
        # Reset index if out of bounds
        if self.current_index >= len(self.training_data):
            self.current_index = max(0, len(self.training_data) - 1)
    
    def get_current_item(self):
        """Get current training data item"""
        if not self.training_data or self.current_index >= len(self.training_data):
            return None
        return self.training_data[self.current_index]
    
    def create_header_panel(self) -> Panel:
        """Create header panel with stats and controls"""
        stats = self.get_stats()
        
        # Title
        title = Text.assemble(
            ("🧠 ", "bold blue"),
            ("Training Data Browser", "bold white"),
            (" v1.0", "dim white")
        )
        
        # Stats summary - make it more comprehensive like db_cli.py
        stats_text = f"""
📊 Total Q&A Pairs: [bold cyan]{stats['total_data']:,}[/bold cyan]
📰 Articles Used: [bold green]{stats['articles_with_data']}[/bold green]
📈 Avg Pairs/Article: [bold yellow]{stats['avg_pairs']}[/bold yellow]
📝 Total Words: [bold magenta]{stats['total_words']:,}[/bold magenta]
📊 Avg Words/Pair: [bold blue]{stats['avg_words']}[/bold blue]
🔥 Recent (last 100): [bold red]{stats['recent_data']}[/bold red]
        """.strip()
        
        # Current filters
        filters = []
        if self.filter_state:
            filters.append(f"State: {self.filter_state}")
        if self.filter_model:
            filters.append(f"Model: {self.filter_model}")
        if self.search_query:
            filters.append(f"Search: {self.search_query}")
        
        filter_text = " | ".join(filters) if filters else "No filters"
        
        # Controls
        controls = """
[bold]Commands:[/bold] n=Next | p=Previous | f=Filter | s=Search | d=Delete | r=Refresh | q=Quit
[dim]Current:[/dim] [cyan]{}/{}[/cyan] | [dim]Filters:[/dim] {}
        """.format(
            self.current_index + 1 if self.training_data else 0,
            len(self.training_data),
            filter_text
        ).strip()
        
        content = f"{title}\n\n{stats_text}\n\n{controls}"
        
        return Panel(
            content,
            box=box.DOUBLE,
            style="blue",
            padding=(1, 2)
        )
    
    def create_item_panel(self) -> Panel:
        """Create panel for current training data item"""
        item = self.get_current_item()
        
        if not item:
            return Panel(
                "[yellow]No training data found[/yellow]\n\n[dim]Try adjusting your filters or adding some training data[/dim]",
                title="📝 Training Data",
                box=box.ROUNDED,
                style="yellow"
            )
        
        uuid, question, answer, state, article_uuid, master_model = item
        
        # Format question (wrap text)
        formatted_question = "\n".join(textwrap.wrap(question, width=80)) if question else "[dim]No question[/dim]"
        
        # Format answer (wrap text)
        formatted_answer = "\n".join(textwrap.wrap(answer, width=80)) if answer else "[dim]No answer[/dim]"
        
        content = f"""
[bold cyan]UUID:[/bold cyan] {uuid[:36]}
[bold green]State:[/bold green] {state}
[bold yellow]Model:[/bold yellow] {master_model}
[bold magenta]Article UUID:[/bold magenta] {article_uuid[:36] if article_uuid else 'None'}

[bold white]Question:[/bold white]
{formatted_question}

[bold white]Answer:[/bold white]
{formatted_answer}
        """.strip()
        
        return Panel(
            content,
            title=f"📝 Training Data ({self.current_index + 1}/{len(self.training_data)})",
            box=box.ROUNDED,
            style="white"
        )
    
    def create_sidebar_panel(self) -> Panel:
        """Create sidebar with models and states"""
        stats = self.get_stats()
        
        # Models breakdown
        models_text = ""
        if stats['models']:
            models_text = "\n".join([
                f"[cyan]{model}[/cyan]: {count}"
                for model, count in sorted(stats['models'].items())
            ])
        else:
            models_text = "[dim]No models found[/dim]"
        
        # States breakdown  
        states_text = ""
        if stats['states']:
            states_text = "\n".join([
                f"[green]{state}[/green]: {count}"
                for state, count in sorted(stats['states'].items())
            ])
        else:
            states_text = "[dim]No states found[/dim]"
        
        content = f"""
[bold]🤖 Models:[/bold]
{models_text}

[bold]📊 States:[/bold]
{states_text}

[bold]🔧 Actions:[/bold]
[cyan]n[/cyan] - Next item
[cyan]p[/cyan] - Previous item
[cyan]f[/cyan] - Filter by state/model
[cyan]s[/cyan] - Search Q&A pairs
[cyan]c[/cyan] - Clear filters
[cyan]d[/cyan] - Delete current item
[cyan]r[/cyan] - Refresh data
[cyan]q[/cyan] - Quit browser
        """.strip()
        
        return Panel(
            content,
            title="📋 Info & Controls",
            box=box.ROUNDED,
            style="blue"
        )
    
    def create_layout(self) -> Layout:
        """Create the main layout"""
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="body")
        )
        
        # Split body into main content and sidebar
        layout["body"].split_row(
            Layout(name="main", ratio=3),
            Layout(name="sidebar", ratio=1)
        )
        
        # Assign panels to layout sections
        layout["header"].update(self.create_header_panel())
        layout["main"].update(self.create_item_panel())
        layout["sidebar"].update(self.create_sidebar_panel())
        
        return layout
    
    def handle_filter(self):
        """Handle filter input"""
        console.print("\n[bold blue]Filter Options:[/bold blue]")
        console.print("1. Filter by state")
        console.print("2. Filter by model")
        console.print("3. Clear filters")
        
        choice = Prompt.ask("Choose option", choices=["1", "2", "3"], default="3")
        
        if choice == "1":
            # Get available states
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT DISTINCT state FROM training_data")
            states = [row[0] for row in c.fetchall()]
            conn.close()
            
            if states:
                console.print(f"\nAvailable states: {', '.join(states)}")
                state = Prompt.ask("Enter state to filter by", default="")
                if state and state in states:
                    self.filter_state = state
                    console.print(f"[green]✓ Filtered by state: {state}[/green]")
                else:
                    console.print("[yellow]Invalid state or filter cleared[/yellow]")
            else:
                console.print("[yellow]No states found[/yellow]")
                
        elif choice == "2":
            # Get available models
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT DISTINCT master_model FROM training_data")
            models = [row[0] for row in c.fetchall()]
            conn.close()
            
            if models:
                console.print(f"\nAvailable models:")
                for i, model in enumerate(models, 1):
                    console.print(f"{i}. {model}")
                
                model_choice = Prompt.ask(f"Select model (1-{len(models)}) or press Enter to cancel", default="")
                if model_choice and model_choice.isdigit():
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(models):
                        self.filter_model = models[idx]
                        console.print(f"[green]✓ Filtered by model: {models[idx]}[/green]")
                    else:
                        console.print("[yellow]Invalid selection[/yellow]")
                else:
                    console.print("[yellow]Filter cancelled[/yellow]")
            else:
                console.print("[yellow]No models found[/yellow]")
                
        elif choice == "3":
            self.filter_state = None
            self.filter_model = None
            self.search_query = None
            console.print("[green]✓ All filters cleared[/green]")
        
        self.load_training_data()
        time.sleep(1)
    
    def handle_search(self):
        """Handle search input"""
        query = Prompt.ask("\n[cyan]Enter search query (or press Enter to clear)[/cyan]", default="")
        if query:
            self.search_query = query
            console.print(f"[green]✓ Searching for: '{query}'[/green]")
        else:
            self.search_query = None
            console.print("[green]✓ Search cleared[/green]")
        
        self.load_training_data()
        time.sleep(1)
    
    def handle_delete(self):
        """Handle delete current item"""
        item = self.get_current_item()
        if not item:
            console.print("[yellow]No item to delete[/yellow]")
            time.sleep(1)
            return
        
        uuid, question, _, _, _, _ = item
        
        # Show confirmation
        console.print(f"\n[red]Delete training data?[/red]")
        console.print(f"[dim]UUID: {uuid}[/dim]")
        console.print(f"[dim]Question: {question[:50]}...[/dim]")
        
        if Confirm.ask("Are you sure?", default=False):
            delete_training_data(uuid)
            console.print("[green]✓ Training data deleted[/green]")
            self.load_training_data()
        else:
            console.print("[yellow]Deletion cancelled[/yellow]")
        
        time.sleep(1)
    
    def render_frame(self):
        """Render a single frame"""
        console.clear()
        console.print(self.create_layout())
        console.print()  # Add some space before prompt
    
    def run_interactive_browser(self):
        """Run the interactive browser with frame-based rendering"""
        console.clear()
        console.print("[green]Loading training data...[/green]")
        self.load_training_data()
        
        if not self.training_data:
            console.print("[yellow]No training data found in database[/yellow]")
            console.print("[dim]Use the data generator to create some training data first[/dim]")
            return
        
        console.print(f"[green]✓ Loaded {len(self.training_data)} training data items[/green]")
        time.sleep(1)
        
        while self.running:
            try:
                # Render current frame
                self.render_frame()
                
                # Get user command
                command = Prompt.ask(
                    "[bold cyan]Command[/bold cyan]",
                    choices=["n", "p", "f", "s", "c", "d", "r", "q", "first", "last"],
                    default="n",
                    show_choices=False
                ).lower().strip()
                
                # Handle commands
                if command == "n":  # Next
                    if self.current_index < len(self.training_data) - 1:
                        self.current_index += 1
                    else:
                        console.print("[yellow]Already at last item[/yellow]")
                        time.sleep(0.5)
                        
                elif command == "p":  # Previous
                    if self.current_index > 0:
                        self.current_index -= 1
                    else:
                        console.print("[yellow]Already at first item[/yellow]")
                        time.sleep(0.5)
                        
                elif command == "first":  # Go to first
                    self.current_index = 0
                    console.print("[green]✓ Moved to first item[/green]")
                    time.sleep(0.5)
                    
                elif command == "last":  # Go to last
                    self.current_index = len(self.training_data) - 1
                    console.print("[green]✓ Moved to last item[/green]")
                    time.sleep(0.5)
                    
                elif command == "f":  # Filter
                    self.handle_filter()
                    
                elif command == "s":  # Search
                    self.handle_search()
                    
                elif command == "c":  # Clear filters
                    self.filter_state = None
                    self.filter_model = None
                    self.search_query = None
                    console.print("[green]✓ All filters cleared[/green]")
                    self.load_training_data()
                    time.sleep(0.5)
                    
                elif command == "d":  # Delete
                    self.handle_delete()
                    
                elif command == "r":  # Refresh
                    console.print("[green]Refreshing data...[/green]")
                    self.load_training_data()
                    time.sleep(0.5)
                    
                elif command == "q":  # Quit
                    self.running = False
                    break
                    
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                time.sleep(1)
        
        console.clear()
        console.print("[yellow]Training Data Browser closed. Goodbye! 👋[/yellow]")

cli = TrainingDataCLI()

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """🧠 Training Data CLI - Browse and manage your training data"""
    if ctx.invoked_subcommand is None:
        # Show banner and stats, then start interactive browser
        console.clear()
        banner = Panel(
            Text.assemble(
                ("🧠 ", "bold blue"),
                ("Training Data CLI", "bold white"),
                (" v1.0", "dim white")
            ),
            box=box.DOUBLE,
            style="blue"
        )
        console.print(Align.center(banner))
        console.print()
        
        stats = cli.get_stats()
        if stats['total_data'] > 0:
            console.print(f"[green]Found {stats['total_data']} training data items[/green]")
            console.print("[dim]Starting interactive browser...[/dim]")
            time.sleep(1)
            cli.run_interactive_browser()
        else:
            console.print("[yellow]No training data found in database[/yellow]")
            console.print("[dim]Use the data generator to create some training data first[/dim]")

@main.command()
def browse():
    """🔍 Start interactive browser"""
    cli.run_interactive_browser()

@main.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed statistics')
def stats(detailed):
    """📊 Show database statistics"""
    stats = cli.get_stats()
    
    if detailed:
        # Detailed view
        console.print("\n" + "="*80)
        console.print(Panel(
            Text("🧠 Training Data Database - Detailed Statistics", style="bold blue"),
            box=box.DOUBLE,
            style="blue"
        ))
        
        # Core Statistics
        core_stats = f"""
[bold cyan]📊 Core Statistics[/bold cyan]
├─ Total Q&A Pairs: [bold white]{stats['total_data']:,}[/bold white]
├─ Valid Pairs (with content): [bold green]{stats['valid_pairs']:,}[/bold green]
├─ Articles Used: [bold yellow]{stats['articles_with_data']:,}[/bold yellow]
├─ Average Pairs per Article: [bold magenta]{stats['avg_pairs']}[/bold magenta]
└─ Recent Data (last 100): [bold red]{stats['recent_data']:,}[/bold red]
        """.strip()
        
        # Content Statistics
        content_stats = f"""
[bold cyan]📝 Content Statistics[/bold cyan]
├─ Total Words: [bold white]{stats['total_words']:,}[/bold white]
├─ Total Characters: [bold green]{stats['total_chars']:,}[/bold green]
├─ Average Words per Pair: [bold yellow]{stats['avg_words']}[/bold yellow]
├─ Average Question Length: [bold magenta]{stats['avg_question_length']} words[/bold magenta]
└─ Average Answer Length: [bold red]{stats['avg_answer_length']} words[/bold red]
        """.strip()
        
        # Model Distribution
        model_stats = "[bold cyan]🤖 Model Distribution[/bold cyan]\n"
        if stats['models']:
            total_model_items = sum(stats['models'].values())
            for model, count in sorted(stats['models'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_model_items) * 100 if total_model_items > 0 else 0
                bar_length = int(percentage / 5)  # Scale bar
                bar = "█" * bar_length + "░" * (20 - bar_length)
                model_stats += f"├─ [cyan]{model}[/cyan]: [white]{count:,}[/white] ([yellow]{percentage:.1f}%[/yellow]) {bar}\n"
        else:
            model_stats += "└─ [dim]No models found[/dim]\n"
        
        # State Distribution
        state_stats = "[bold cyan]📊 State Distribution[/bold cyan]\n"
        if stats['states']:
            total_state_items = sum(stats['states'].values())
            for state, count in sorted(stats['states'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_state_items) * 100 if total_state_items > 0 else 0
                bar_length = int(percentage / 5)  # Scale bar
                bar = "█" * bar_length + "░" * (20 - bar_length)
                state_stats += f"├─ [green]{state}[/green]: [white]{count:,}[/white] ([yellow]{percentage:.1f}%[/yellow]) {bar}\n"
        else:
            state_stats += "└─ [dim]No states found[/dim]\n"
        
        # Display in panels
        console.print(Columns([
            Panel(core_stats, title="📊 Core", box=box.ROUNDED, style="blue"),
            Panel(content_stats, title="📝 Content", box=box.ROUNDED, style="green")
        ], equal=True))
        
        console.print("\n")
        console.print(Columns([
            Panel(model_stats.rstrip(), title="🤖 Models", box=box.ROUNDED, style="cyan"),
            Panel(state_stats.rstrip(), title="📊 States", box=box.ROUNDED, style="yellow")
        ], equal=True))
        
    else:
        # Simple view (existing code)
        panels = []
        
        # Main stats
        main_stats = f"""
📊 Total Q&A Pairs: [bold cyan]{stats['total_data']:,}[/bold cyan]
📰 Articles Used: [bold green]{stats['articles_with_data']}[/bold green]
📈 Avg Pairs/Article: [bold yellow]{stats['avg_pairs']}[/bold yellow]
📝 Total Words: [bold magenta]{stats['total_words']:,}[/bold magenta]
📊 Avg Words/Pair: [bold blue]{stats['avg_words']}[/bold blue]
🔥 Recent (last 100): [bold red]{stats['recent_data']}[/bold red]
        """.strip()
        
        panels.append(Panel(main_stats, title="📊 Statistics", box=box.ROUNDED))
        
        # Models breakdown
        if stats['models']:
            models_info = "\n".join([
                f"[cyan]{model}[/cyan]: {count:,}" 
                for model, count in sorted(stats['models'].items(), key=lambda x: x[1], reverse=True)
            ])
        else:
            models_info = "[dim]No models found[/dim]"
            
        panels.append(Panel(models_info, title="🤖 Models", box=box.ROUNDED))
        
        # States breakdown
        if stats['states']:
            states_info = "\n".join([
                f"[green]{state}[/green]: {count:,}" 
                for state, count in sorted(stats['states'].items(), key=lambda x: x[1], reverse=True)
            ])
        else:
            states_info = "[dim]No states found[/dim]"
            
        panels.append(Panel(states_info, title="📊 States", box=box.ROUNDED))
        
        console.print(Columns(panels, equal=True))
        
        console.print(f"\n[dim]💡 Use --detailed for comprehensive statistics[/dim]")

@main.command()
@click.argument('query')
@click.option('--limit', '-n', default=10, help='Maximum number of results')
def search(query, limit):
    """🔍 Search training data"""
    results = fuzzy_search(query, limit)
    
    if not results:
        console.print(f"[yellow]No results found for '{query}'[/yellow]")
        return
    
    table = Table(title=f"🔍 Search Results for '{query}'", box=box.ROUNDED)
    table.add_column("UUID", style="cyan", width=12)
    table.add_column("Question", style="white", width=40)
    table.add_column("Answer", style="green", width=40)
    table.add_column("Model", style="yellow", width=15)
    
    for result in results:
        uuid, question, answer, state, article_uuid, master_model = result
        # Truncate long text
        question_short = question[:37] + "..." if len(question) > 40 else question
        answer_short = answer[:37] + "..." if len(answer) > 40 else answer
        
        table.add_row(
            uuid[:12],
            question_short,
            answer_short,
            master_model
        )
    
    console.print(table)

if __name__ == '__main__':
    main()