#!/usr/bin/env python3
"""
Documentation Crawler for Shader Languages

This script crawls official documentation sites for various shader languages,
extracts information about functions, types, operators, and other features,
and stores the structured data for further analysis.
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests
import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from markdownify import markdownify
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Setup logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crossgl-docs/docs/docs_crawler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("docs_crawler")

# Ensure cache directories exist
os.makedirs(".cache/docs", exist_ok=True)
os.makedirs(".cache/analysis", exist_ok=True)
os.makedirs(".github/logs", exist_ok=True)

# Load NLP models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    logger.warning("Spacy model not found, downloading...")
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Documentation sources configuration
DOCUMENTATION_SOURCES = {
    "metal": {
        "base_url": (
            "https://developer.apple.com/documentation/metal/metal_shading_language"
        ),
        "function_paths": [
            "/mathematical_functions",
            "/relational_functions",
            "/geometric_functions",
            "/texture_functions",
        ],
        "type_paths": [
            "/data_types",
            "/vector_and_matrix_data_types",
        ],
        "operator_paths": [
            "/operators",
        ],
        "function_selector": "a.link.symbol-name",
        "dynamic_loading": True,
        "requires_js": True,
        "authentication": False,
    },
    "directx": {
        "base_url": "https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl",
        "function_paths": [
            "/dx-graphics-hlsl-intrinsic-functions",
        ],
        "type_paths": [
            "/dx-graphics-hlsl-data-types",
        ],
        "operator_paths": [
            "/dx-graphics-hlsl-operators",
        ],
        "function_selector": (
            'h2[id^="intrinsic-functions"], h3[id^="intrinsic-functions"]'
        ),
        "dynamic_loading": False,
        "requires_js": False,
        "authentication": False,
    },
    "opengl": {
        "base_url": "https://www.khronos.org/registry/OpenGL-Refpages/gl4/",
        "function_paths": [
            "html/indexflat.php",
        ],
        "type_paths": [
            "html/glsl.html",
        ],
        "operator_paths": [
            "html/glsl.html",
        ],
        "function_selector": "li a",
        "dynamic_loading": False,
        "requires_js": False,
        "authentication": False,
    },
    "vulkan": {
        "base_url": (
            "https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html"
        ),
        "function_paths": [
            "#spirvenv-capabilities-shaderintrinsics",
        ],
        "type_paths": [
            "#spirvenv-types",
        ],
        "operator_paths": [
            "#spirvenv-operators",
        ],
        "function_selector": "code",
        "dynamic_loading": False,
        "requires_js": False,
        "authentication": False,
    },
    "slang": {
        "base_url": "https://shader-slang.com/slang/docs/reference/",
        "function_paths": [
            "stdlib-math.html",
            "stdlib-vector.html",
            "stdlib-matrix.html",
        ],
        "type_paths": [
            "types.html",
        ],
        "operator_paths": [
            "operators.html",
        ],
        "function_selector": "code",
        "dynamic_loading": False,
        "requires_js": False,
        "authentication": False,
    },
    "mojo": {
        "base_url": "https://docs.modular.com/mojo/stdlib",
        "function_paths": [
            "builtin.html",
            "math.html",
        ],
        "type_paths": [
            "dtype.html",
        ],
        "operator_paths": [
            "operator.html",
        ],
        "function_selector": "h3.function",
        "dynamic_loading": True,
        "requires_js": True,
        "authentication": False,
    },
}


class DocCrawler:
    """Base class for document crawlers"""

    def __init__(self, language: str):
        self.language = language
        self.config = DOCUMENTATION_SOURCES[language]
        self.cache_dir = Path(f".cache/docs/{language}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir = Path(".cache/analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Setup browser if needed
        self.browser = None
        if self.config["requires_js"]:
            self._setup_browser()

    def _setup_browser(self):
        """Initialize a headless browser for JavaScript rendering"""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        service = Service(ChromeDriverManager().install())
        self.browser = webdriver.Chrome(service=service, options=options)

    def _get_page(self, url: str) -> str:
        """Get HTML content from a URL, using browser or requests as needed"""
        cache_file = self.cache_dir / f"{hash(url)}.html"

        # Check if cached version exists
        if cache_file.exists():
            logger.info(f"Using cached version of {url}")
            return cache_file.read_text(encoding="utf-8")

        logger.info(f"Fetching {url}")

        if self.config["requires_js"]:
            self.browser.get(url)
            # Wait for page to load
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Handle dynamic loading if needed
            if self.config["dynamic_loading"]:
                # Scroll to load more content
                self.browser.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )
                time.sleep(2)

            html = self.browser.page_source
        else:
            response = requests.get(url)
            response.raise_for_status()
            html = response.text

        # Cache the result
        cache_file.write_text(html, encoding="utf-8")
        return html

    def extract_functions(self) -> List[Dict[str, Any]]:
        """Extract function information from documentation"""
        all_functions = []

        for path in self.config["function_paths"]:
            url = self.config["base_url"] + path
            html = self._get_page(url)
            soup = BeautifulSoup(html, "html.parser")

            # Extract function elements based on language-specific configuration
            function_elements = soup.select(self.config["function_selector"])

            for element in function_elements:
                function_info = self._parse_function_element(element)
                if function_info:
                    all_functions.append(function_info)

        return all_functions

    def extract_types(self) -> List[Dict[str, Any]]:
        """Extract type information from documentation"""
        all_types = []

        for path in self.config["type_paths"]:
            url = self.config["base_url"] + path
            html = self._get_page(url)
            soup = BeautifulSoup(html, "html.parser")

            # Type extraction is language-specific, implemented in subclasses
            all_types.extend(self._parse_type_page(soup))

        return all_types

    def extract_operators(self) -> List[Dict[str, Any]]:
        """Extract operator information from documentation"""
        all_operators = []

        for path in self.config["operator_paths"]:
            url = self.config["base_url"] + path
            html = self._get_page(url)
            soup = BeautifulSoup(html, "html.parser")

            # Operator extraction is language-specific, implemented in subclasses
            all_operators.extend(self._parse_operator_page(soup))

        return all_operators

    def _parse_function_element(self, element) -> Optional[Dict[str, Any]]:
        """Parse a function element to extract name, signature, description, etc."""
        # Base implementation, should be overridden in subclasses
        try:
            function_name = element.get_text().strip()
            if not function_name:
                return None

            # Get parent element for potential description
            parent = element.parent
            description = ""
            if parent:
                # Look for a paragraph after the function name
                next_sibling = parent.find_next_sibling("p")
                if next_sibling:
                    description = next_sibling.get_text().strip()

            return {
                "name": function_name,
                "signature": function_name,
                "description": description,
                "parameters": [],
                "return_type": "",
                "url": element.get("href") if element.name == "a" else "",
                "source": self.language,
            }
        except Exception as e:
            logger.error(f"Error parsing function element: {e}")
            return None

    def _parse_type_page(self, soup) -> List[Dict[str, Any]]:
        """Parse a type documentation page"""
        # Base implementation, should be overridden in subclasses
        return []

    def _parse_operator_page(self, soup) -> List[Dict[str, Any]]:
        """Parse an operator documentation page"""
        # Base implementation, should be overridden in subclasses
        return []

    def download_all(self):
        """Download all documentation pages for caching"""
        urls = []

        # Collect all URLs to download
        for path_list in [
            self.config["function_paths"],
            self.config["type_paths"],
            self.config["operator_paths"],
        ]:
            for path in path_list:
                urls.append(self.config["base_url"] + path)

        # Download pages in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self._get_page, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    future.result()
                    logger.info(f"Downloaded {url}")
                except Exception as e:
                    logger.error(f"Error downloading {url}: {e}")

    def analyze(self):
        """Analyze documentation and save structured data"""
        logger.info(f"Analyzing {self.language} documentation...")

        data = {
            "language": self.language,
            "timestamp": datetime.now().isoformat(),
            "functions": self.extract_functions(),
            "types": self.extract_types(),
            "operators": self.extract_operators(),
        }

        # Save to JSON file
        output_file = self.analysis_dir / f"{self.language}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Analysis complete. Saved to {output_file}")
        return data

    def close(self):
        """Close browser if used"""
        if self.browser:
            self.browser.quit()


# Language-specific crawler implementations


class MetalCrawler(DocCrawler):
    """Metal Shading Language documentation crawler"""

    def _parse_function_element(self, element) -> Optional[Dict[str, Any]]:
        try:
            # Metal documentation has a specific structure
            function_name = element.get_text().strip()

            # Find the description
            description_div = element.find_parent("div").find_next_sibling("div")
            description = description_div.get_text().strip() if description_div else ""

            # Try to find the function signature
            signature = function_name
            signature_element = element.find_parent("div").find("code")
            if signature_element:
                signature = signature_element.get_text().strip()

            # Parse parameters and return type from signature
            parameters = []
            return_type = ""

            # Regex to extract return type and parameters
            signature_match = re.search(r"(\w+)\s+(\w+)\((.*?)\)", signature)
            if signature_match:
                return_type = signature_match.group(1)
                parameters_str = signature_match.group(3)

                # Parse parameters
                if parameters_str:
                    param_parts = parameters_str.split(",")
                    for part in param_parts:
                        part = part.strip()
                        if part:
                            param_type, param_name = part.rsplit(" ", 1)
                            parameters.append(
                                {"name": param_name, "type": param_type.strip()}
                            )

            return {
                "name": function_name,
                "signature": signature,
                "description": description,
                "parameters": parameters,
                "return_type": return_type,
                "url": (
                    self.config["base_url"] + element.get("href")
                    if element.name == "a" and element.get("href")
                    else ""
                ),
                "source": self.language,
            }
        except Exception as e:
            logger.error(f"Error parsing Metal function element: {e}")
            return None

    def _parse_type_page(self, soup) -> List[Dict[str, Any]]:
        types = []
        # Look for type definitions in data_types page
        type_elements = soup.select("h3.jump-target")

        for elem in type_elements:
            type_name = elem.get_text().strip()
            description_elem = elem.find_next_sibling("p")
            description = (
                description_elem.get_text().strip() if description_elem else ""
            )

            types.append(
                {"name": type_name, "description": description, "source": self.language}
            )

        return types

    def _parse_operator_page(self, soup) -> List[Dict[str, Any]]:
        operators = []
        # Find operator tables
        operator_tables = soup.select("table")

        for table in operator_tables:
            # Get table headers
            headers = [th.get_text().strip() for th in table.select("th")]

            # Process each row
            for row in table.select("tr"):
                cells = row.select("td")
                if not cells:
                    continue

                operator_data = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        operator_data[headers[i].lower()] = cell.get_text().strip()

                if "operator" in operator_data:
                    operators.append(
                        {
                            "symbol": operator_data.get("operator", ""),
                            "name": operator_data.get(
                                "name", operator_data.get("operator", "")
                            ),
                            "description": operator_data.get("description", ""),
                            "example": operator_data.get("example", ""),
                            "source": self.language,
                        }
                    )

        return operators


class DirectXCrawler(DocCrawler):
    """DirectX HLSL documentation crawler"""

    def _parse_function_element(self, element) -> Optional[Dict[str, Any]]:
        try:
            function_name = element.get_text().strip()

            # Skip non-function headers
            if not re.match(r"^[a-zA-Z0-9_]+$", function_name):
                return None

            # Find description
            description = ""
            next_elem = element.find_next_sibling()
            if next_elem and next_elem.name == "p":
                description = next_elem.get_text().strip()

            # Try to find code example with signature
            code_elem = None
            curr_elem = element
            for _ in range(5):  # Check next 5 siblings
                curr_elem = curr_elem.find_next_sibling()
                if curr_elem and curr_elem.name == "pre":
                    code_elem = curr_elem
                    break

            signature = function_name
            parameters = []
            return_type = ""

            if code_elem:
                code_text = code_elem.get_text().strip()
                # Parse the signature
                sig_match = re.search(r"(\w+)\s+(\w+)\s*\((.*?)\)", code_text)
                if sig_match:
                    return_type = sig_match.group(1)
                    function_name = sig_match.group(2)  # Update function name
                    signature = code_text

                    # Parse parameters
                    params_text = sig_match.group(3)
                    if params_text:
                        param_parts = params_text.split(",")
                        for part in param_parts:
                            part = part.strip()
                            if part:
                                param_match = re.match(r"(\w+)\s+(\w+)", part)
                                if param_match:
                                    param_type = param_match.group(1)
                                    param_name = param_match.group(2)
                                    parameters.append(
                                        {"name": param_name, "type": param_type}
                                    )

            return {
                "name": function_name,
                "signature": signature,
                "description": description,
                "parameters": parameters,
                "return_type": return_type,
                "url": (
                    f"{self.config['base_url']}/dx-graphics-hlsl-intrinsic-functions#{function_name.lower()}"
                ),
                "source": self.language,
            }
        except Exception as e:
            logger.error(f"Error parsing DirectX function element: {e}")
            return None

    def _parse_type_page(self, soup) -> List[Dict[str, Any]]:
        types = []
        # Find type sections
        type_sections = soup.select("h3")

        for section in type_sections:
            section_id = section.get("id", "")
            if "type" in section_id.lower():
                type_name = section.get_text().strip()
                description = ""

                # Get description from next paragraph
                next_elem = section.find_next_sibling("p")
                if next_elem:
                    description = next_elem.get_text().strip()

                types.append(
                    {
                        "name": type_name,
                        "description": description,
                        "source": self.language,
                    }
                )

        return types

    def _parse_operator_page(self, soup) -> List[Dict[str, Any]]:
        operators = []
        # Find operator tables
        operator_sections = soup.select("h2")

        for section in operator_sections:
            if "operator" in section.get_text().lower():
                # Find tables in this section
                tables = section.find_next_siblings("table")

                for table in tables:
                    rows = table.select("tr")
                    if not rows:
                        continue

                    # Get headers from first row
                    headers = [
                        th.get_text().strip().lower() for th in rows[0].select("th")
                    ]

                    # Process data rows
                    for row in rows[1:]:
                        cells = row.select("td")
                        if not cells:
                            continue

                        operator_data = {}
                        for i, cell in enumerate(cells):
                            if i < len(headers):
                                operator_data[headers[i]] = cell.get_text().strip()

                        operators.append(
                            {
                                "symbol": operator_data.get("operator", ""),
                                "name": operator_data.get(
                                    "name", operator_data.get("operator", "")
                                ),
                                "description": operator_data.get("description", ""),
                                "example": operator_data.get("example", ""),
                                "source": self.language,
                            }
                        )

        return operators


class OpenGLCrawler(DocCrawler):
    """OpenGL GLSL documentation crawler"""

    def _parse_function_element(self, element) -> Optional[Dict[str, Any]]:
        try:
            if not element.name == "a":
                return None

            href = element.get("href", "")
            if not href or not href.endswith(".xml"):
                return None

            function_name = element.get_text().strip()

            # Follow the link to get function details
            function_url = (
                f"{self.config['base_url']}/html/{href.replace('.xml', '.html')}"
            )
            function_html = self._get_page(function_url)
            function_soup = BeautifulSoup(function_html, "html.parser")

            # Extract function details
            synopsis = function_soup.select_one(".refsynopsisdiv")
            signature = ""
            if synopsis:
                signature_elem = synopsis.select_one("pre.funcsynopsis")
                if signature_elem:
                    signature = signature_elem.get_text().strip()

            # Extract description
            description = ""
            desc_div = function_soup.select_one(".refsect1 .para")
            if desc_div:
                description = desc_div.get_text().strip()

            # Parse parameters and return type
            parameters = []
            return_type = ""

            if signature:
                # Parse return type and function name
                sig_match = re.search(r"(\w+)\s+(\w+)\s*\((.*?)\)", signature)
                if sig_match:
                    return_type = sig_match.group(1)
                    function_name = sig_match.group(2)
                    params_text = sig_match.group(3)

                    # Parse parameters
                    if params_text and params_text != "void":
                        param_parts = params_text.split(",")
                        for part in param_parts:
                            part = part.strip()
                            if part:
                                param_match = re.match(r"(\w+)\s+(\w+)", part)
                                if param_match:
                                    param_type = param_match.group(1)
                                    param_name = param_match.group(2)
                                    parameters.append(
                                        {"name": param_name, "type": param_type}
                                    )

            return {
                "name": function_name,
                "signature": signature,
                "description": description,
                "parameters": parameters,
                "return_type": return_type,
                "url": function_url,
                "source": self.language,
            }
        except Exception as e:
            logger.error(f"Error parsing OpenGL function element: {e}")
            return None

    def _parse_type_page(self, soup) -> List[Dict[str, Any]]:
        types = []
        # Find data type sections
        type_sections = soup.select("h2, h3")

        current_section = ""
        for section in type_sections:
            section_text = section.get_text().lower()
            if "data type" in section_text:
                current_section = section.get_text().strip()
                continue

            if current_section and section.name == "h3":
                type_name = section.get_text().strip()

                # Get description
                description = ""
                next_elem = section.find_next_sibling("p")
                if next_elem:
                    description = next_elem.get_text().strip()

                types.append(
                    {
                        "name": type_name,
                        "description": description,
                        "category": current_section,
                        "source": self.language,
                    }
                )

        return types

    def _parse_operator_page(self, soup) -> List[Dict[str, Any]]:
        operators = []
        # Find operator sections
        operator_sections = soup.select("h2, h3")

        in_operator_section = False
        for section in operator_sections:
            section_text = section.get_text().lower()

            # Check if we're in the operators section
            if "operator" in section_text and section.name == "h2":
                in_operator_section = True
                continue
            elif section.name == "h2":
                in_operator_section = False

            if in_operator_section and section.name == "h3":
                operator_name = section.get_text().strip()

                # Get description
                description = ""
                next_elem = section.find_next_sibling("p")
                if next_elem:
                    description = next_elem.get_text().strip()

                # Extract operator symbol from name or description
                symbol = ""
                symbol_match = re.search(
                    r"(\+|\-|\*|\/|\%|\=|\!|\<|\>|\&|\||\^|\~)+", operator_name
                )
                if symbol_match:
                    symbol = symbol_match.group(0)

                operators.append(
                    {
                        "symbol": symbol,
                        "name": operator_name,
                        "description": description,
                        "source": self.language,
                    }
                )

        return operators


# Factory method to get the appropriate crawler
def get_crawler(language: str) -> DocCrawler:
    """Get a language-specific crawler instance"""
    if language not in DOCUMENTATION_SOURCES:
        raise ValueError(f"Unsupported language: {language}")

    if language == "metal":
        return MetalCrawler(language)
    elif language == "directx":
        return DirectXCrawler(language)
    elif language == "opengl":
        return OpenGLCrawler(language)
    else:
        # Use the base crawler for other languages
        return DocCrawler(language)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Documentation Crawler for Shader Languages"
    )
    parser.add_argument("--language", "-l", required=True, help="Language to analyze")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download documentation without analysis",
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze without downloading"
    )

    args = parser.parse_args()

    if args.language not in DOCUMENTATION_SOURCES:
        logger.error(f"Unsupported language: {args.language}")
        return 1

    crawler = get_crawler(args.language)

    try:
        if args.download_only:
            logger.info(f"Downloading documentation for {args.language}...")
            crawler.download_all()
        elif args.analyze_only:
            logger.info(f"Analyzing documentation for {args.language}...")
            crawler.analyze()
        else:
            # Both download and analyze
            logger.info(f"Processing documentation for {args.language}...")
            crawler.download_all()
            crawler.analyze()
    finally:
        crawler.close()

    return 0


if __name__ == "__main__":
    exit(main())
