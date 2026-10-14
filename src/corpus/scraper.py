"""
scraper.py
==========
Downloads DevOps/cloud documentation from publicly available sources.

Strategy per source:
  GitHub-hosted (Docker, Kubernetes, GitHub Actions):
    -> Clone the repo (shallow clone) and parse .md / .mdx files.
  Web-hosted (Nginx, PostgreSQL, man7.org):
    -> Follow sitemap.xml and scrape HTML pages.

All scraped pages are saved as raw files in corpus/raw/<source>/
so the corpus can be rebuilt from scratch without re-downloading.

Usage:
  scraper = Scraper(config, raw_dir="corpus/raw")
  pages = scraper.scrape_all()
  # pages is List[RawPage]
"""

import os
import re
import time
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datetime import date
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model for a scraped page (before cleaning)
# ---------------------------------------------------------------------------

@dataclass
class RawPage:
    source:      str       # e.g. 'docker'
    topic:       str       # e.g. 'networking'
    version:     str       # e.g. '26.0' or 'latest'
    url:         str       # Canonical URL
    date_scraped: str      # ISO date
    content:     str       # Raw HTML or Markdown text
    content_type: str      # 'html' or 'markdown'


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

SOURCES = {
    "docker": {
        "type": "github",
        "repo_url": "https://github.com/docker/docs",
        "branch": "main",
        "file_glob": "**/*.md",
        "topic_map": {           # directory -> topic label
            "network": "networking",
            "storage": "storage",
            "compose": "compose",
            "engine":  "engine",
            "build":   "build",
            "desktop": "desktop",
            "security":"security",
        },
        "default_topic": "general",
        "version": "latest",
        "base_url": "https://docs.docker.com",
    },
    "kubernetes": {
        "type": "github",
        "repo_url": "https://github.com/kubernetes/website",
        "branch": "main",
        "file_glob": "content/en/docs/**/*.md",
        "topic_map": {
            "concepts": "concepts",
            "tasks":    "tasks",
            "tutorials":"tutorials",
            "reference":"reference",
            "setup":    "setup",
        },
        "default_topic": "general",
        "version": "latest",
        "base_url": "https://kubernetes.io/docs",
    },
    "nginx": {
        "type": "web_sitemap",
        "sitemap_url": "https://nginx.org/en/docs/",
        "base_url": "https://nginx.org",
        "allowed_path_prefix": "/en/docs",
        "version": "latest",
        "default_topic": "general",
        "topic_map": {
            "http":    "http",
            "mail":    "mail",
            "stream":  "stream",
            "njs":     "scripting",
        },
    },
    "postgresql": {
        "type": "web_sitemap",
        "sitemap_url": "https://www.postgresql.org/docs/current/",
        "base_url": "https://www.postgresql.org",
        "allowed_path_prefix": "/docs/current",
        "version": "current",
        "default_topic": "general",
        "topic_map": {
            "sql":       "sql",
            "admin":     "administration",
            "backup":    "backup",
            "high-availability": "high_availability",
            "monitoring":"monitoring",
        },
    },
    "linux_commands": {
        "type": "web_sitemap",
        "sitemap_url": "https://man7.org/linux/man-pages/index.html",
        "base_url": "https://man7.org",
        "allowed_path_prefix": "/linux/man-pages",
        "version": "latest",
        "default_topic": "commands",
        "topic_map": {
            "man1": "user_commands",
            "man5": "file_formats",
            "man8": "admin_commands",
        },
    },
    "github_actions": {
        "type": "github",
        "repo_url": "https://github.com/github/docs",
        "branch": "main",
        "file_glob": "content/actions/**/*.md",
        "topic_map": {
            "using-workflows": "workflows",
            "writing-workflows":"writing",
            "managing-workflow-runs":"runs",
            "hosting-your-own-runners":"runners",
        },
        "default_topic": "general",
        "version": "latest",
        "base_url": "https://docs.github.com/actions",
    },
    "helm": {
        "type": "github",
        "repo_url": "https://github.com/helm/helm-www",
        "branch": "main",
        "file_glob": "content/en/docs/**/*.md",
        "topic_map": {
            "chart_template_guide": "templating",
            "helm": "cli",
            "intro": "intro",
            "topics": "concepts",
        },
        "default_topic": "helm",
        "version": "v3",
        "base_url": "https://helm.sh/docs",
    },
    "prometheus": {
        "type": "github",
        "repo_url": "https://github.com/prometheus/docs",
        "branch": "main",
        "file_glob": "content/docs/**/*.md",
        "topic_map": {
            "alerting": "alerting",
            "querying": "promql",
            "instrumenting": "instrumentation",
            "operating": "operations",
            "introduction": "concepts",
        },
        "default_topic": "monitoring",
        "version": "latest",
        "base_url": "https://prometheus.io/docs",
    },
    "fastapi": {
        "type": "github",
        "repo_url": "https://github.com/fastapi/fastapi",
        "branch": "master",
        "file_glob": "docs/en/docs/**/*.md",
        "topic_map": {
            "tutorial": "tutorial",
            "advanced": "advanced",
            "async": "async",
        },
        "default_topic": "api",
        "version": "latest",
        "base_url": "https://fastapi.tiangolo.com",
    },
}


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class Scraper:
    """
    Downloads documentation pages from all configured sources.

    Args:
        raw_dir:        Where to save raw downloaded files.
        sources:        Dict of source definitions (defaults to SOURCES).
        delay_seconds:  Polite delay between web requests.
        max_pages:      Max pages per source (None = no limit).
        skip_existing:  If True, skip sources already downloaded in raw_dir.
    """

    def __init__(
        self,
        raw_dir: str = "corpus/raw",
        sources: dict = None,
        delay_seconds: float = 0.5,
        max_pages: Optional[int] = None,
        skip_existing: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.sources = sources or SOURCES
        self.delay = delay_seconds
        self.max_pages = max_pages
        self.skip_existing = skip_existing
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        self.today = date.today().isoformat()

    # ── Main entry-point ───────────────────────────────────────────────────

    def scrape_all(self) -> List[RawPage]:
        """Scrape every configured source. Returns all raw pages."""
        all_pages: List[RawPage] = []
        for source_name, cfg in self.sources.items():
            print(f"\n{'='*60}")
            print(f"  Scraping: {source_name}")
            print(f"{'='*60}")

            source_dir = self.raw_dir / source_name
            source_dir.mkdir(parents=True, exist_ok=True)

            has_files = False
            if (source_dir / "_clone").exists() and any((source_dir / "_clone").glob("**/*.md")):
                has_files = True
            elif any(f.is_file() for f in source_dir.iterdir() if f.name != "_clone"):
                has_files = True

            if self.skip_existing and has_files:
                print(f"  -> Skipping (already downloaded). Delete {source_dir} to re-download.")
                pages = self._load_cached(source_name, source_dir, cfg)
            elif cfg["type"] == "github":
                pages = self._scrape_github(source_name, cfg, source_dir)
            elif cfg["type"] == "web_sitemap":
                pages = self._scrape_web(source_name, cfg, source_dir)
            else:
                log.warning(f"Unknown source type: {cfg['type']}")
                pages = []

            print(f"  -> {len(pages)} pages collected from {source_name}")
            all_pages.extend(pages)

        print(f"\n{'='*60}")
        print(f"  TOTAL PAGES SCRAPED: {len(all_pages)}")
        print(f"{'='*60}")
        return all_pages

    # ── GitHub source ──────────────────────────────────────────────────────

    def _scrape_github(
        self, source_name: str, cfg: dict, save_dir: Path
    ) -> List[RawPage]:
        """
        Shallow-clone the GitHub repo and walk the local files.

        Using `git clone --depth 1` is orders of magnitude faster than
        individual GitHub API calls (one pack download vs. N HTTP requests).
        Requires git to be installed (standard on most systems).
        Falls back to ZIP download if git is unavailable.
        """
        import subprocess

        repo_url  = cfg["repo_url"]
        branch    = cfg.get("branch", "main")
        file_glob = cfg.get("file_glob", "**/*.md")
        parts     = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]

        clone_dir = save_dir / "_clone"

        # ── Strategy 1: git clone (fastest) ──────────────────────────────
        git_ok = self._try_git_clone(repo_url, branch, clone_dir)

        # ── Strategy 2: ZIP download fallback ────────────────────────────
        if not git_ok:
            print(f"  -> git not available - using ZIP download fallback")
            zip_ok = self._try_zip_download(owner, repo, branch, clone_dir)
            if not zip_ok:
                log.error(f"Cannot download {source_name} via git or ZIP.")
                return []

        # ── Walk local clone for matching files ───────────────────────────
        pages: List[RawPage] = []
        all_md = list(clone_dir.rglob("*.md")) + list(clone_dir.rglob("*.mdx"))

        # Apply glob pattern filter if it's more specific than **/*.md
        if file_glob not in ("**/*.md", "**/*.mdx"):
            clean_glob = file_glob.replace("\\", "/")
            prefix = clean_glob.split("**")[0].strip("/")
            if prefix:
                all_md = [
                    f for f in all_md
                    if prefix.lower() in str(f.relative_to(clone_dir)).replace("\\", "/").lower()
                ]

        if self.max_pages:
            all_md = all_md[: self.max_pages]

        print(f"  -> Found {len(all_md)} Markdown files in local clone")

        for fpath in tqdm(all_md, desc=f"  Parsing {source_name}"):
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                log.warning(f"  Cannot read {fpath}: {e}")
                continue

            rel_path = str(fpath.relative_to(clone_dir)).replace("\\", "/")
            topic    = self._infer_topic(rel_path, cfg)
            url      = f"{cfg['base_url']}/{rel_path}"

            pages.append(RawPage(
                source=source_name,
                topic=topic,
                version=cfg.get("version", "latest"),
                url=url,
                date_scraped=self.today,
                content=content,
                content_type="markdown",
            ))

        return pages

    def _try_git_clone(self, repo_url: str, branch: str, clone_dir: Path) -> bool:
        """Shallow clone using git. Returns True on success."""
        import subprocess
        if clone_dir.exists() and any(clone_dir.glob("**/*.md")):
            print(f"  -> Using existing local clone at {clone_dir}")
            return True
        clone_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "git", "clone",
            "--depth", "1",
            "--branch", branch,
            "--single-branch",
            "--no-tags",
            repo_url,
            str(clone_dir),
        ]
        print(f"  -> git clone --depth 1 {repo_url} ...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"  -> Clone complete")
                return True
            else:
                log.warning(f"git clone failed: {result.stderr[:200]}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.warning(f"git not found or timed out: {e}")
            return False

    def _try_zip_download(
        self, owner: str, repo: str, branch: str, extract_dir: Path
    ) -> bool:
        """Download the repo as a ZIP from GitHub. Returns True on success."""
        import zipfile, io
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        print(f"  -> Downloading ZIP: {zip_url}")
        try:
            r = self.session.get(zip_url, timeout=120, stream=True)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            data  = b""
            for chunk in tqdm(
                r.iter_content(chunk_size=1024 * 1024),
                total=max(1, total // (1024 * 1024)),
                unit="MB", desc="  Downloading"
            ):
                data += chunk

            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(extract_dir)

            # GitHub ZIPs extract to <repo>-<branch>/ subdirectory - flatten
            subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                inner = subdirs[0]
                for item in inner.iterdir():
                    item.rename(extract_dir / item.name)
                inner.rmdir()

            print(f"  -> ZIP extracted to {extract_dir}")
            return True
        except Exception as e:
            log.error(f"ZIP download failed: {e}")
            return False

    # ── Web / sitemap source ───────────────────────────────────────────────

    def _scrape_web(
        self, source_name: str, cfg: dict, save_dir: Path
    ) -> List[RawPage]:
        """Crawl a documentation site starting from its index page."""
        base_url    = cfg["base_url"]
        start_url   = cfg["sitemap_url"]
        allowed_pfx = cfg.get("allowed_path_prefix", "")

        # Collect URLs to visit
        urls_to_visit = self._collect_links(start_url, base_url, allowed_pfx)
        if self.max_pages:
            urls_to_visit = urls_to_visit[: self.max_pages]

        print(f"  -> Found {len(urls_to_visit)} pages to scrape")
        pages: List[RawPage] = []

        for url in tqdm(urls_to_visit, desc=f"  Scraping {source_name}"):
            time.sleep(self.delay)
            try:
                r = self.session.get(url, timeout=20)
                r.raise_for_status()
                content = r.text
            except Exception as e:
                log.warning(f"  Failed: {url} - {e}")
                continue

            topic = self._infer_topic(url, cfg)
            safe_name = hashlib.md5(url.encode()).hexdigest()[:16] + ".html"
            (save_dir / safe_name).write_text(content, encoding="utf-8")

            pages.append(RawPage(
                source=source_name,
                topic=topic,
                version=cfg.get("version", "latest"),
                url=url,
                date_scraped=self.today,
                content=content,
                content_type="html",
            ))

        return pages

    def _collect_links(
        self, start_url: str, base_url: str, allowed_prefix: str
    ) -> List[str]:
        """Return all unique doc links found on the index page."""
        try:
            r = self.session.get(start_url, timeout=20)
            r.raise_for_status()
        except Exception as e:
            log.error(f"Cannot load index {start_url}: {e}")
            return [start_url]

        soup = BeautifulSoup(r.text, "lxml")
        links: List[str] = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(base_url, href)
            parsed = urlparse(full)

            if parsed.netloc != urlparse(base_url).netloc:
                continue
            if allowed_prefix and not parsed.path.startswith(allowed_prefix):
                continue
            if full in seen:
                continue
            if any(full.endswith(ext) for ext in [".png", ".jpg", ".pdf", ".zip"]):
                continue

            seen.add(full)
            links.append(full)

        return links

    # ── Cache loading (re-read already-downloaded raw files) ───────────────

    def _load_cached(
        self, source_name: str, source_dir: Path, cfg: dict
    ) -> List[RawPage]:
        clone_dir = source_dir / "_clone"
        if clone_dir.exists() and any(clone_dir.glob("**/*.md")):
            return self._scrape_github(source_name, cfg, source_dir)

        pages: List[RawPage] = []
        for fpath in source_dir.iterdir():
            if not fpath.is_file():
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                ct = "markdown" if fpath.suffix in (".md", ".mdx") else "html"
                pages.append(RawPage(
                    source=source_name,
                    topic=cfg.get("default_topic", "general"),
                    version=cfg.get("version", "latest"),
                    url=cfg.get("base_url", ""),
                    date_scraped=self.today,
                    content=content,
                    content_type=ct,
                ))
            except Exception as e:
                log.warning(f"Cannot read cached file {fpath}: {e}")
        return pages

    # ── Topic inference ────────────────────────────────────────────────────

    def _infer_topic(self, path_or_url: str, cfg: dict) -> str:
        """Map a file path or URL to a topic label using the source's topic_map."""
        topic_map = cfg.get("topic_map", {})
        for keyword, topic in topic_map.items():
            if keyword.lower() in path_or_url.lower():
                return topic
        return cfg.get("default_topic", "general")
