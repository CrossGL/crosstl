#!/usr/bin/env python3
"""
CrossTL Command Line Interface.
Professional CLI for universal programming language translation.
"""

import argparse
import sys
import os
import crosstl


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="crosstl",
        description="CrossTL - Universal Programming Language Translator",
        epilog="""
Examples:
  crosstl shader.cu metal                          # Translate CUDA to Metal
  crosstl -s cuda -t opengl input.cu output.glsl  # Explicit languages
  crosstl --info                                   # Show system info
  crosstl --list-languages                        # List supported languages
  
Supported Languages:
  cuda, metal, directx, opengl, vulkan, rust, mojo, hip, slang, crossgl
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument("input", nargs="?", help="Input file path or source code")
    parser.add_argument(
        "target",
        nargs="?",
        help="Target language (cuda, metal, directx, opengl, vulkan, rust, mojo, hip, slang)",
    )

    # Optional arguments
    parser.add_argument(
        "-s", "--source", help="Source language (auto-detected if not specified)"
    )
    parser.add_argument(
        "-o", "--output", help="Output file path (auto-generated if not specified)"
    )

    # Information commands
    parser.add_argument(
        "--info", action="store_true", help="Show CrossTL system information"
    )
    parser.add_argument(
        "--list-languages", action="store_true", help="List all supported languages"
    )
    parser.add_argument(
        "--list-extensions", action="store_true", help="List supported file extensions"
    )

    # Options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--version", action="version", version=f"CrossTL {crosstl.__version__}"
    )
    parser.add_argument(
        "--check-syntax", action="store_true", help="Check syntax without translation"
    )

    return parser


def handle_info_commands(args):
    """Handle information display commands."""
    if args.info:
        crosstl.info()
        return True

    if args.list_languages:
        languages = crosstl.get_supported_languages()
        print("Supported Languages:")
        print("===================")
        for lang in sorted(languages):
            extensions = crosstl.SUPPORTED_LANGUAGES[lang]["extensions"]
            print(f"  {lang:<10} - {', '.join(extensions)}")
        return True

    if args.list_extensions:
        print("Supported File Extensions:")
        print("=========================")
        for lang, info in sorted(crosstl.SUPPORTED_LANGUAGES.items()):
            for ext in info["extensions"]:
                print(f"  {ext:<8} - {lang}")
        return True

    return False


def validate_arguments(args):
    """Validate command line arguments."""
    if not args.input or not args.target:
        print(
            "Error: Both input and target language are required for translation",
            file=sys.stderr,
        )
        return False

    # Validate languages
    supported = crosstl.get_supported_languages()

    if args.source and args.source not in supported:
        print(f"Error: Unsupported source language '{args.source}'", file=sys.stderr)
        print(f"Supported languages: {', '.join(sorted(supported))}", file=sys.stderr)
        return False

    if args.target not in supported:
        print(f"Error: Unsupported target language '{args.target}'", file=sys.stderr)
        print(f"Supported languages: {', '.join(sorted(supported))}", file=sys.stderr)
        return False

    return True


def check_syntax_only(input_file, source_language, verbose):
    """Check syntax without performing translation."""
    try:
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                source_code = f.read()

            if source_language is None:
                source_language = crosstl.detect_language(input_file, source_code)
        else:
            source_code = input_file
            if source_language is None:
                source_language = crosstl.detect_language("temp.code", source_code)

        # Get lexer and parser
        source_info = crosstl.SUPPORTED_LANGUAGES[source_language]

        # Tokenize
        lexer = source_info["lexer"](source_code)
        tokens = lexer.tokenize()

        if verbose:
            print(f"Tokenization successful: {len(tokens)} tokens")

        # Parse
        parser = source_info["parser"](tokens)
        parser.parse()

        if verbose:
            print(f"Parsing successful: AST generated")

        print(f"✓ Syntax check passed for {source_language}")
        return True

    except Exception as e:
        print(f"✗ Syntax error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


def perform_translation(args):
    """Perform the actual translation."""
    try:
        start_time = None
        if args.verbose:
            import time

            start_time = time.time()
            print(f"Starting translation: {args.input} -> {args.target}")

        # Perform translation
        result = crosstl.translate(args.input, args.target, args.source, args.output)

        if args.verbose and start_time:
            elapsed = time.time() - start_time
            print(f"Translation completed in {elapsed:.3f}s")

        # Output results
        if args.output:
            print(f"✓ Translation successful: {args.input} -> {args.output}")

            if args.verbose:
                size = len(result)
                print(f"Generated {size} characters of {args.target} code")
        else:
            print(result)

        return True

    except Exception as e:
        print(f"✗ Translation failed: {e}", file=sys.stderr)

        if args.verbose:
            import traceback

            traceback.print_exc()

        return False


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle information commands
    if handle_info_commands(args):
        return 0

    # Handle syntax checking
    if args.check_syntax:
        if not args.input:
            print("Error: Input file required for syntax checking", file=sys.stderr)
            return 1

        success = check_syntax_only(args.input, args.source, args.verbose)
        return 0 if success else 1

    # Validate arguments for translation
    if not validate_arguments(args):
        parser.print_help()
        return 1

    # Perform translation
    success = perform_translation(args)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
