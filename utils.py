#!/usr/bin/env python3
"""
🎨 COLORPRINT CLASS FOR BEAUTIFUL OUTPUT
Part of Make It Heavy - Async Performance Edition

This module provides the centralized ColorPrint class that delivers
beautiful, colorful console output across all async components.
Features cross-platform support and consistent emoji-rich styling.

Usage: from utils import ColorPrint
"""

from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)


class ColorPrint:
    """Colorful printing utility for better UX"""

    @staticmethod
    def info(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.CYAN}ℹ️  {message}{Style.RESET_ALL}")

    @staticmethod
    def success(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

    @staticmethod
    def warning(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")

    @staticmethod
    def error(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

    @staticmethod
    def processing(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.BLUE}🔄 {message}{Style.RESET_ALL}")

    @staticmethod
    def tool(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.MAGENTA}🔧 {message}{Style.RESET_ALL}")

    @staticmethod
    def cache(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.YELLOW}🎯 {message}{Style.RESET_ALL}")

    @staticmethod
    def metrics(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.CYAN}📊 {message}{Style.RESET_ALL}")

    @staticmethod
    def performance(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.GREEN}⚡ {message}{Style.RESET_ALL}")

    @staticmethod
    def debug(message: str, silent: bool = False):
        if not silent:
            print(f"{Fore.WHITE}{Style.DIM}🔍 {message}{Style.RESET_ALL}")

    @staticmethod
    def header(message: str, silent: bool = False):
        if not silent:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 60}")
            print(f"🚀 {message}")
            print(f"{'=' * 60}{Style.RESET_ALL}")

    @staticmethod
    def subheader(message: str, silent: bool = False):
        if not silent:
            print(f"\n{Fore.BLUE}{Style.BRIGHT}--- {message} ---{Style.RESET_ALL}")
