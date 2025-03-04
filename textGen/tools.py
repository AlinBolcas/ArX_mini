import sys
import os
import datetime
import qrcode
import requests
import webbrowser
import html2text
import smtplib
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from typing import Optional, Any, Dict, List, Union

# Adjust sys.path to include the parent directory where 'utils' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import Utils  # Import shared utilities

# Attempt to import OAI
OAI_module = Utils.import_file("oai.py")
OAI = OAI_module.OAI if OAI_module else None

# Initialize OpenAI integration
oai_instance = OAI(api_keys_path=None) if OAI else None


class Tools:
    """
    LLM-accessible tools for file handling, web crawling, visualization, AI utilities, and system operations.
    
    These tools provide structured functionality that the LLM can call dynamically.
    """

    @staticmethod
    def save_named_file(content: str, name: Optional[str] = None, extension: str = "py") -> str:
        """
        Save content to a file with a specified or AI-generated name.

        Args:
            content (str): The text content to save in the file.
            name (Optional[str]): The desired filename without extension (if None, a name is auto-generated).
            extension (str): The file extension, default is "py".

        Returns:
            str: The full file path where the content was saved.
        """
        return Utils.save_file(content, name, extension)

    @staticmethod
    def import_tool_module(name: str) -> Optional[Any]:
        """Dynamically locate and import a Python module by its filename."""
        return Utils.import_file(name)

    @staticmethod
    def get_codebase_snapshot() -> Dict[str, Any]:
        """
        Generate a structured snapshot of the current project’s codebase.

        This function retrieves the codebase snapshot using Utils.get_codebase_snapshot(),
        which scans the project directory and returns a structured dictionary containing
        filenames, file sizes, imports, and a brief preview of each code file.

        Returns:
            Dict[str, Any]: JSON object representing the directory structure, file details, and code summaries.
        """
        return Utils.get_codebase_snapshot()

    @staticmethod
    def test_code_syntax(code: str) -> str:
        """
        Test the syntax of a given Python code snippet.

        This function checks if the provided Python code has valid syntax and returns any errors found.

        Args:
            code (str): The Python code to be checked.

        Returns:
            str: Output of the syntax test or error details.
        """
        return Utils.test_code(code)

    @staticmethod
    def web_crawl_url(url: str, convert_to_markdown: bool = True) -> str:
        """
        Retrieve and extract readable text content from a given webpage.

        Args:
            url (str): The webpage URL to fetch and extract text from.
            convert_to_markdown (bool): If True, converts HTML content into Markdown format.

        Returns:
            str: The extracted text from the webpage (Markdown if enabled, plain text otherwise).
        """
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            plain_text = soup.get_text(separator="\n", strip=True)

            if convert_to_markdown:
                converter = html2text.HTML2Text()
                converter.ignore_links = False
                return converter.handle(response.text).strip()
            return plain_text
        except requests.RequestException as e:
            return f"Error fetching URL: {str(e)}"

    @staticmethod
    def web_crawl_query(query: str, convert_to_markdown: bool = True) -> Dict[str, Any]:
        """
        Retrieve and extract readable text content from the top 5 valid webpages related to a search query.

        Args:
            query (str): The search query to generate relevant web links.
            convert_to_markdown (bool): If True, converts HTML content into Markdown format.

        Returns:
            Dict[str, Any]: A dictionary containing accessible links as keys and their extracted text content as values.
        """
        if not oai_instance:
            return {"error": "OpenAI module not initialized."}

        # Step 1: Generate potential links using OpenAI
        try:
            structured_output = oai_instance.structured_output(
                user_prompt=f"Find 5 real, publicly accessible web links for information on: '{query}'. "
                            "Only include URLs from reputable sources like Wikipedia, research papers, tech blogs, or news sites. "
                            "Exclude login-required pages, dead links, placeholders, or pages with minimal content.",
                system_prompt="Return only a structured JSON list of working URLs. Do not include descriptions or explanations.",
            )
        except Exception as e:
            return {"error": f"OpenAI API error: {str(e)}"}

        # Validate output
        if not isinstance(structured_output, list) or not structured_output:
            return {"error": "Failed to retrieve valid links from OpenAI."}

        valid_links = {}

        # Step 2: Validate and scrape each link
        for url in structured_output:
            try:
                print(f"Checking URL: {url}")

                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                plain_text = soup.get_text(separator="\n", strip=True)

                if convert_to_markdown:
                    converter = html2text.HTML2Text()
                    converter.ignore_links = False
                    page_content = converter.handle(response.text).strip()
                else:
                    page_content = plain_text

                if page_content:
                    valid_links[url] = page_content[:500]  # Limit content length
                    print(f"✔ Successfully retrieved content from {url}")

            except requests.RequestException as e:
                print(f"✖ Skipping broken link: {url} - Error: {e}")

        return valid_links if valid_links else {"error": "No accessible links found."}

    @staticmethod
    def open_url_in_browser(url: str) -> None:
        """Open a specified URL in the user's default web browser."""
        webbrowser.open(url)

    @staticmethod
    def get_current_datetime() -> str:
        """Retrieve the current date and time in ISO 8601 format."""
        return datetime.datetime.now().isoformat()

    @staticmethod
    def generate_qr_code(data: str, output_file: str = "qr_code.png") -> str:
        """
        Generate a QR code image from a given text input.

        Args:
            data (str): The data (URL or text) to encode in the QR code.
            output_file (str): The output filename for the QR image.

        Returns:
            str: The file path where the QR code image was saved.
        """
        qr = qrcode.make(data)
        output_path = os.path.join(Utils.get_output_path(), output_file)
        qr.save(output_path)
        return output_path

    @staticmethod
    def summarize_text(content: str, max_length: int = 100) -> str:
        """
        Summarize a given text to a specified max length.

        Args:
            content (str): The input text to summarize.
            max_length (int): The maximum number of words for the summary.

        Returns:
            str: The summarized text.
        """
        if not oai_instance:
            return "Error: OpenAI module not initialized."

        return oai_instance.chat_completion(f"Summarize the following text in {max_length} words:\n{content}")

    @staticmethod
    def translate_text(text: str, target_language: str) -> str:
        """
        Translate a given text into a specified language.

        Args:
            text (str): The input text to translate.
            target_language (str): The target language (e.g., "French", "Japanese").

        Returns:
            str: The translated text.
        """
        if not oai_instance:
            return "Error: OpenAI module not initialized."

        return oai_instance.chat_completion(f"Translate the following text into {target_language}:\n{text}")

    @staticmethod
    def send_email(recipient: str, subject: str, message: str) -> bool:
        """
        Send an email using an SMTP service.

        Args:
            recipient (str): The recipient's email address.
            subject (str): The subject of the email.
            message (str): The email body content.

        Returns:
            bool: True if the email was successfully sent, False otherwise.
        """
        SMTP_SERVER = "smtp.gmail.com"
        SMTP_PORT = 587
        EMAIL_USER = "info@arvolve.ai"
        EMAIL_PASS = "cqux vlzo lthv avyn"  # Replace with a secure App Password

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            email_message = f"Subject: {subject}\n\n{message}"
            server.sendmail(EMAIL_USER, recipient, email_message)
            server.quit()
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    @staticmethod
    def convert_text_to_speech(text: str, output_file: str = "speech.mp3") -> str:
        """
        Convert text into speech and save it as an audio file using OpenAI's TTS.

        Args:
            text (str): The input text to convert to speech.
            output_file (str): The filename for the generated audio file.

        Returns:
            str: The file path where the speech audio file was saved.
        """
        try:
            if not oai_instance:
                raise ValueError("OpenAI instance is not initialized.")

            output_dir = Utils.get_output_path()
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            output_path = os.path.join(output_dir, output_file)

            tts_result = oai_instance.text_to_speech(text, output_path=output_path)

            if isinstance(tts_result, str) and os.path.exists(output_path):
                return output_path
            else:
                raise RuntimeError("Failed to generate speech output.")
        except Exception as e:
            print(f"Error in convert_text_to_speech: {e}")
            return ""

    @staticmethod
    def generate_image(prompt: str, style: Optional[str] = "realistic", output_file: str = "generated_image.png") -> str:
        """
        Generate an AI-generated image using OpenAI's API and save it locally.

        Args:
            prompt (str): The description of the image to generate.
            style (Optional[str]): The preferred image style.
            output_file (str): The filename for the generated image.

        Returns:
            str: The file path where the generated image was saved.
        """
        try:
            if not oai_instance:
                raise ValueError("OpenAI instance is not initialized.")

            # Fetch image URL from OpenAI
            image_urls = oai_instance.generate_image(prompt=prompt, style=style)
            if not image_urls:
                raise ValueError("OpenAI image generation failed.")

            image_url = image_urls[0]  # Get first image

            # Download image and save it
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            output_dir = Utils.get_output_path()
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
            output_path = os.path.join(output_dir, output_file)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            return output_path
        except Exception as e:
            print(f"Error in generate_image: {e}")
            return ""

    @staticmethod
    def plot_data(data: Dict[str, Any], chart_type: str = "bar", output_file: str = "chart.png") -> str:
        """
        Generate a data visualization chart based on provided input.

        Args:
            data (Dict[str, Any]): JSON object containing "labels" (list) and "values" (list).
            chart_type (str): The type of chart to generate ("bar", "pie", "line", "scatter").
            output_file (str): The filename for the saved chart.

        Returns:
            str: The file path where the chart image was saved.
        """
        labels = data.get("labels", [])
        values = data.get("values", [])
        if not labels or not values:
            raise ValueError("Data must contain 'labels' and 'values' keys.")

        plt.figure(figsize=(8, 5))
        if chart_type == "bar":
            plt.bar(labels, values, color="blue")
            plt.xlabel("Labels")
            plt.ylabel("Values")
            plt.title("Bar Chart")
        elif chart_type == "pie":
            plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
            plt.title("Pie Chart")
        elif chart_type == "line":
            plt.plot(labels, values, marker="o", linestyle="-", color="red")
            plt.xlabel("Labels")
            plt.ylabel("Values")
            plt.title("Line Chart")
        elif chart_type == "scatter":
            plt.scatter(labels, values, color="green")
            plt.xlabel("Labels")
            plt.ylabel("Values")
            plt.title("Scatter Plot")
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        output_path = os.path.join(Utils.get_output_path(), output_file)
        plt.savefig(output_path)
        plt.close()

        return output_path
            
if __name__ == '__main__':
    print("=== Testing Tools Module Functions ===\n")

    # 1️⃣ Test: Save Named File
    print("Test 1: save_named_file")
    try:
        file_path = Tools.save_named_file("This is a test content", "test_file", extension="txt")
        print(f"✅ File saved at: {file_path}\n")
    except Exception as e:
        print(f"❌ save_named_file encountered an error: {e}\n")

    # 2️⃣ Test: Import Tool Module
    print("Test 2: import_tool_module")
    try:
        imported_module = Tools.import_tool_module("oai.py")
        print(f"✅ Successfully imported: {imported_module.__name__}\n") if imported_module else print("❌ oai.py not found or failed to import.\n")
    except Exception as e:
        print(f"❌ import_tool_module encountered an error: {e}\n")

    # 3️⃣ Test: Get Codebase Snapshot
    print("Test 3: get_codebase_snapshot")
    try:
        snapshot = Tools.get_codebase_snapshot()
        print(f"✅ Codebase snapshot retrieved successfully with {len(snapshot)} files.\n") if snapshot else print("❌ No codebase snapshot retrieved.\n")
    except Exception as e:
        print(f"❌ get_codebase_snapshot encountered an error: {e}\n")

    # 4️⃣ Test: Test Code Syntax
    print("Test 4: test_code_syntax")
    test_code = "def test_func():\n    return 42"
    try:
        syntax_result = Tools.test_code_syntax(test_code)
        print(f"✅ Syntax test result: {syntax_result}\n")
    except Exception as e:
        print(f"❌ test_code_syntax encountered an error: {e}\n")

    # 5️⃣ Test: Web Crawling - Retrieve Content from a URL
    print("Test 5: web_crawl_url")
    test_url = "https://www.python.org/"
    try:
        page_content = Tools.web_crawl_url(test_url)
        print(f"✅ Successfully retrieved content (first 300 chars):\n{page_content[:300]}...\n") if page_content else print(f"⚠️ No meaningful content extracted from {test_url}.\n")
    except Exception as e:
        print(f"❌ web_crawl_url encountered an error: {e}\n")

    # 6️⃣ Test: Web Query Crawling
    print("Test 6: web_crawl_query")
    search_query = "latest AI advancements"
    try:
        results = Tools.web_crawl_query(search_query)
        print(f"✅ Successfully retrieved content from search query '{search_query}'.\n") if results else print(f"⚠️ No valid links found.\n")
    except Exception as e:
        print(f"❌ web_crawl_query encountered an error: {e}\n")

    # 7️⃣ Test: Open URL in Browser
    print("Test 7: open_url_in_browser")
    test_browser_url = "https://www.google.com/"
    Tools.open_url_in_browser(test_browser_url)
    print("✅ Browser should now be open.\n")

    # 8️⃣ Test: Get Current Datetime
    print("Test 8: get_current_datetime")
    try:
        current_time = Tools.get_current_datetime()
        print(f"✅ Current datetime: {current_time}\n")
    except Exception as e:
        print(f"❌ get_current_datetime encountered an error: {e}\n")

    # 9️⃣ Test: Generate QR Code
    print("Test 9: generate_qr_code")
    try:
        qr_path = Tools.generate_qr_code("https://www.example.com", "test_qr.png")
        print(f"✅ QR Code generated at: {qr_path}\n")
    except Exception as e:
        print(f"❌ generate_qr_code encountered an error: {e}\n")

    # 🔟 Test: Summarize Text
    print("Test 10: summarize_text")
    test_text = "Artificial intelligence simulates human intelligence processes."
    try:
        summary = Tools.summarize_text(test_text, max_length=10)
        print(f"✅ Summary: {summary}\n")
    except Exception as e:
        print(f"❌ summarize_text encountered an error: {e}\n")

    # 1️⃣1️⃣ Test: Translate Text
    print("Test 11: translate_text")
    try:
        translated_text = Tools.translate_text("Hello, how are you?", "French")
        print(f"✅ Translated Text: {translated_text}\n")
    except Exception as e:
        print(f"❌ translate_text encountered an error: {e}\n")

    # 1️⃣2️⃣ Test: Convert Text to Speech
    print("Test 12: convert_text_to_speech")
    try:
        tts_path = Tools.convert_text_to_speech("Hello, this is a test speech!", "test_speech.mp3")
        print(f"✅ Speech generated at: {tts_path}\n")
    except Exception as e:
        print(f"❌ convert_text_to_speech encountered an error: {e}\n")

    # 1️⃣3️⃣ Test: Generate AI Image
    print("Test 13: generate_ai_image")
    try:
        image_path = Tools.generate_image("A futuristic city with flying cars", style="realistic", output_file="test_image.png")
        print(f"✅ AI-generated image saved at: {image_path}\n")
    except Exception as e:
        print(f"❌ generate_ai_image encountered an error: {e}\n")

    # 1️⃣4️⃣ Test: Plot Data
    print("Test 14: plot_data")
    try:
        sample_data = {"labels": ["Jan", "Feb", "Mar"], "values": [100, 150, 200]}
        chart_path = Tools.plot_data(sample_data, chart_type="bar", output_file="test_chart.png")
        print(f"✅ Chart generated at: {chart_path}\n")
    except Exception as e:
        print(f"❌ plot_data encountered an error: {e}\n")

    # 1️⃣5️⃣ Test: Send Email
    print("Test 15: send_email")
    recipient_email = "abolcas@gmail.com"  # Replace with a real email for actual test
    email_subject = "Test Email from Tools.py"
    email_message = "This is a test email sent from the Tools module."

    user_email_confirmation = input("Attempt to send test email? (y/n): ").strip().lower()
    if user_email_confirmation == "y":
        try:
            email_status = Tools.send_email(recipient_email, email_subject, email_message)
            print("✅ Test email sent successfully.\n") if email_status else print("❌ Failed to send test email.\n")
        except Exception as e:
            print(f"❌ send_email encountered an error: {e}\n")
    else:
        print("⚠️ Email test skipped.\n")

    print("=== All tests completed ===")
