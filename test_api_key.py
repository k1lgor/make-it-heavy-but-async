#!/usr/bin/env python3
"""
🔑 API KEY TESTER
Quick script to test if your OpenRouter API key is working
"""

import yaml
import requests
import json
from utils import ColorPrint


def test_api_key():
    """Test the OpenRouter API key"""
    try:
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        api_key = config["openrouter"]["api_key"]
        base_url = config["openrouter"]["base_url"]

        ColorPrint.info("Testing OpenRouter API key...")
        ColorPrint.info(f"API Key: {api_key[:8]}...{api_key[-8:]}")
        ColorPrint.info(f"Base URL: {base_url}")

        # Test with a simple request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Simple test payload
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": 'Say "API key test successful"'}],
            "max_tokens": 50,
        }

        ColorPrint.processing("Making test API call...")

        response = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30
        )

        if response.status_code == 200:
            ColorPrint.success("✅ API key is working!")
            result = response.json()
            message = result["choices"][0]["message"]["content"]
            ColorPrint.success(f"Response: {message}")
            return True
        else:
            ColorPrint.error(f"❌ API call failed with status {response.status_code}")
            ColorPrint.error(f"Response: {response.text}")

            # Specific error messages
            if response.status_code == 401:
                ColorPrint.warning("🔑 This means your API key is invalid or expired")
                ColorPrint.info("Solutions:")
                ColorPrint.info("1. Check your API key at https://openrouter.ai/keys")
                ColorPrint.info("2. Make sure you have credits in your account")
                ColorPrint.info("3. Generate a new API key if needed")
            elif response.status_code == 429:
                ColorPrint.warning(
                    "⏰ Rate limit exceeded - wait a moment and try again"
                )
            elif response.status_code == 402:
                ColorPrint.warning(
                    "💳 Insufficient credits - add credits to your account"
                )

            return False

    except FileNotFoundError:
        ColorPrint.error("❌ config.yaml not found!")
        return False
    except Exception as e:
        ColorPrint.error(f"❌ Test failed: {str(e)}")
        return False


def check_account_info():
    """Check account information and credits"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        api_key = config["openrouter"]["api_key"]

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        ColorPrint.processing("Checking account information...")

        # Check credits
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            ColorPrint.success("📊 Account Information:")
            ColorPrint.info(f"   Label: {data.get('data', {}).get('label', 'N/A')}")
            ColorPrint.info(f"   Usage: ${data.get('data', {}).get('usage', 0):.4f}")
            ColorPrint.info(f"   Limit: ${data.get('data', {}).get('limit', 0):.2f}")

            limit = data.get("data", {}).get("limit", 0)
            usage = data.get("data", {}).get("usage", 0)
            remaining = limit - usage

            if remaining > 0:
                ColorPrint.success(f"💰 Remaining credits: ${remaining:.4f}")
            else:
                ColorPrint.warning("⚠️ No remaining credits - please add funds")

        else:
            ColorPrint.warning(
                f"⚠️ Could not fetch account info: {response.status_code}"
            )

    except Exception as e:
        ColorPrint.warning(f"⚠️ Could not check account info: {str(e)}")


if __name__ == "__main__":
    ColorPrint.header("OPENROUTER API KEY TESTER")

    # Test the API key
    if test_api_key():
        ColorPrint.success("🎉 Your API key is working correctly!")

        # Check account info
        check_account_info()

        ColorPrint.info("\n🚀 You can now use the async agents:")
        ColorPrint.info("   python main_async.py")
        ColorPrint.info("   python make_it_heavy_async.py")
        ColorPrint.info("   streamlit run streamlit_app.py")
    else:
        ColorPrint.error("❌ API key test failed!")
        ColorPrint.info("\n🔧 To fix this:")
        ColorPrint.info("1. Go to https://openrouter.ai/keys")
        ColorPrint.info("2. Create a new API key or check your existing one")
        ColorPrint.info("3. Update config.yaml with your valid API key")
        ColorPrint.info("4. Make sure you have credits in your account")
